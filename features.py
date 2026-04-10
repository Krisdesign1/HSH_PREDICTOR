# ============================================================
#  HSH PREDICTOR — Feature Engineering
# ============================================================

import numpy as np
import pandas as pd
import logging
from database import get_conn
import psycopg2.extras

logger = logging.getLogger(__name__)


def compute_team_hsh_stats(team_id: int, league_id: int,
                            as_home: bool = True, last_n: int = 20) -> dict:
    """
    Calcule les stats HSH d'une équipe sur ses N derniers matchs.
    Retourne un dict de features.
    """
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        field = "home_id" if as_home else "away_id"
        cur.execute(f"""
            SELECT hsh_result, hsh_goals_h1, hsh_goals_h2,
                   ht_home, ht_away, ft_home, ft_away
            FROM matches
            WHERE {field} = %s
              AND league_id = %s
              AND has_ht_data = TRUE
              AND hsh_result IS NOT NULL
            ORDER BY match_date DESC
            LIMIT %s
        """, (team_id, league_id, last_n))

        rows = cur.fetchall()

    if not rows:
        return _empty_stats(as_home)

    n = len(rows)
    h1_wins = sum(1 for r in rows if r["hsh_result"] == "H1")
    h2_wins = sum(1 for r in rows if r["hsh_result"] == "H2")
    eq      = sum(1 for r in rows if r["hsh_result"] == "EQ")

    avg_goals_h1 = np.mean([r["hsh_goals_h1"] for r in rows if r["hsh_goals_h1"] is not None])
    avg_goals_h2 = np.mean([r["hsh_goals_h2"] for r in rows if r["hsh_goals_h2"] is not None])

    prefix = "home" if as_home else "away"
    return {
        f"{prefix}_pct_h1":        h1_wins / n,
        f"{prefix}_pct_h2":        h2_wins / n,
        f"{prefix}_pct_eq":        eq / n,
        f"{prefix}_avg_goals_h1":  float(avg_goals_h1),
        f"{prefix}_avg_goals_h2":  float(avg_goals_h2),
        f"{prefix}_h2_h1_ratio":   float(avg_goals_h2 / max(avg_goals_h1, 0.01)),
        f"{prefix}_n_matches":     n,
    }


def _empty_stats(as_home: bool) -> dict:
    prefix = "home" if as_home else "away"
    return {
        f"{prefix}_pct_h1":        0.33,
        f"{prefix}_pct_h2":        0.33,
        f"{prefix}_pct_eq":        0.33,
        f"{prefix}_avg_goals_h1":  0.7,
        f"{prefix}_avg_goals_h2":  0.9,
        f"{prefix}_h2_h1_ratio":   1.0,
        f"{prefix}_n_matches":     0,
    }


def build_match_features(match: dict, league_group: str) -> dict:
    """
    Construit le vecteur de features pour un match donné.
    Combine les stats domicile + extérieur + contexte ligue.
    """
    home_id   = match["home_id"]
    away_id   = match["away_id"]
    league_id = match["league_id"]

    # Stats des 20 derniers matchs à domicile/extérieur
    home_stats = compute_team_hsh_stats(home_id, league_id, as_home=True,  last_n=20)
    away_stats = compute_team_hsh_stats(away_id, league_id, as_home=False, last_n=20)

    # Stats des 10 derniers (forme récente)
    home_recent = compute_team_hsh_stats(home_id, league_id, as_home=True,  last_n=10)
    away_recent = compute_team_hsh_stats(away_id, league_id, as_home=False, last_n=10)

    # Encodage du groupe de ligue
    group_enc = {"A": 0, "B": 1, "C": 2, "D": 3}.get(league_group, 3)

    # Feature combinée : tendance commune H2
    combined_h2_trend = (
        home_stats["home_pct_h2"] + away_stats["away_pct_h2"]
    ) / 2

    combined_h1_trend = (
        home_stats["home_pct_h1"] + away_stats["away_pct_h1"]
    ) / 2

    # Ratio buts 2H/1H combiné
    combined_ratio = (
        home_stats["home_h2_h1_ratio"] + away_stats["away_h2_h1_ratio"]
    ) / 2

    features = {
        # Stats domicile (20 matchs)
        **home_stats,
        # Stats extérieur (20 matchs)
        **away_stats,
        # Forme récente domicile (10 matchs)
        "home_recent_pct_h1":   home_recent["home_pct_h1"],
        "home_recent_pct_h2":   home_recent["home_pct_h2"],
        # Forme récente extérieur (10 matchs)
        "away_recent_pct_h1":   away_recent["away_pct_h1"],
        "away_recent_pct_h2":   away_recent["away_pct_h2"],
        # Features combinées
        "combined_h2_trend":    combined_h2_trend,
        "combined_h1_trend":    combined_h1_trend,
        "combined_ratio":       combined_ratio,
        # Contexte ligue
        "league_group_enc":     group_enc,
        # PPG (points par match — indicateur force équipe)
        "home_ppg":             match.get("home_ppg") or 1.5,
        "away_ppg":             match.get("away_ppg") or 1.5,
        "ppg_diff":             (match.get("home_ppg") or 1.5) - (match.get("away_ppg") or 1.5),
    }

    return features


def build_training_dataset(league_group: str = None) -> pd.DataFrame:
    """
    Construit le dataset complet pour l'entraînement ML.
    Retourne un DataFrame avec features + label.
    """
    from database import get_matches_for_training

    matches = get_matches_for_training(league_group)
    logger.info(f"📦 {len(matches)} matchs récupérés pour entraînement (groupe={league_group})")

    rows = []
    for i, match in enumerate(matches):
        if i % 500 == 0:
            logger.info(f"  Features: {i}/{len(matches)}")

        lg = match.get("league_group", "D")
        feats = build_match_features(dict(match), lg)
        feats["label"] = match["hsh_result"]  # H1 / H2 / EQ
        rows.append(feats)

    df = pd.DataFrame(rows)
    logger.info(f"✅ Dataset construit : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    logger.info(f"   Distribution HSH : {df['label'].value_counts().to_dict()}")

    return df


# ── Colonnes utilisées par le modèle ────────────────────────
FEATURE_COLUMNS = [
    "home_pct_h1", "home_pct_h2", "home_pct_eq",
    "home_avg_goals_h1", "home_avg_goals_h2", "home_h2_h1_ratio",
    "away_pct_h1", "away_pct_h2", "away_pct_eq",
    "away_avg_goals_h1", "away_avg_goals_h2", "away_h2_h1_ratio",
    "home_recent_pct_h1", "home_recent_pct_h2",
    "away_recent_pct_h1", "away_recent_pct_h2",
    "combined_h2_trend", "combined_h1_trend", "combined_ratio",
    "league_group_enc",
    "home_ppg", "away_ppg", "ppg_diff",
]

LABEL_COLUMN = "label"
LABEL_ENCODER = {"H1": 0, "H2": 1, "EQ": 2}
LABEL_DECODER = {0: "H1", 1: "H2", 2: "EQ"}
