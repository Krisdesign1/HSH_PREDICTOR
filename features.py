# ============================================================
#  HSH PREDICTOR — Feature Engineering
# ============================================================

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import logging
from database import get_conn
import psycopg2.extras

logger = logging.getLogger(__name__)


def _coerce_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value


def _aggregate_team_stats(rows: list[dict], as_home: bool) -> dict:
    if not rows:
        return _empty_stats(as_home)

    n = len(rows)
    h1_wins = sum(1 for r in rows if r["hsh_result"] == "H1")
    h2_wins = sum(1 for r in rows if r["hsh_result"] == "H2")
    eq = sum(1 for r in rows if r["hsh_result"] == "EQ")

    goals_h1 = [r["hsh_goals_h1"] for r in rows if r["hsh_goals_h1"] is not None]
    goals_h2 = [r["hsh_goals_h2"] for r in rows if r["hsh_goals_h2"] is not None]
    avg_goals_h1 = np.mean(goals_h1) if goals_h1 else 0.7
    avg_goals_h2 = np.mean(goals_h2) if goals_h2 else 0.9

    prefix = "home" if as_home else "away"
    return {
        f"{prefix}_pct_h1": h1_wins / n,
        f"{prefix}_pct_h2": h2_wins / n,
        f"{prefix}_pct_eq": eq / n,
        f"{prefix}_avg_goals_h1": float(avg_goals_h1),
        f"{prefix}_avg_goals_h2": float(avg_goals_h2),
        f"{prefix}_h2_h1_ratio": float(avg_goals_h2 / max(avg_goals_h1, 0.01)),
        f"{prefix}_n_matches": n,
    }


class HistoricalStatsCache:
    """Précharge l'historique nécessaire pour éviter les requêtes N+1."""

    def __init__(self, rows_by_key: dict):
        self.rows_by_key = rows_by_key
        self.dates_by_key = {
            key: [row["match_date"] for row in rows]
            for key, rows in rows_by_key.items()
        }

    @classmethod
    def from_matches(cls, matches: list[dict] | list):
        match_dicts = [dict(match) for match in matches]
        league_ids = sorted(
            {
                match.get("league_id")
                for match in match_dicts
                if match.get("league_id") is not None
            }
        )
        team_ids = sorted(
            {
                team_id
                for match in match_dicts
                for team_id in (match.get("home_id"), match.get("away_id"))
                if team_id is not None
            }
        )
        if not league_ids or not team_ids:
            return cls({})

        with get_conn() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                SELECT footystats_id, league_id, home_id, away_id, match_date,
                       hsh_result, hsh_goals_h1, hsh_goals_h2,
                       ht_home, ht_away, ft_home, ft_away
                FROM matches
                WHERE league_id = ANY(%s)
                  AND has_ht_data = TRUE
                  AND hsh_result IS NOT NULL
                  AND (home_id = ANY(%s) OR away_id = ANY(%s))
                ORDER BY match_date ASC
                """,
                (league_ids, team_ids, team_ids),
            )
            records = cur.fetchall()

        rows_by_key = defaultdict(list)
        for record in records:
            row = dict(record)
            row["match_date"] = _coerce_datetime(row.get("match_date"))
            if row["match_date"] is None:
                continue
            rows_by_key[(row["home_id"], row["league_id"], True)].append(row)
            rows_by_key[(row["away_id"], row["league_id"], False)].append(row)

        logger.info(
            "🗃️  Cache historique chargé : %s lignes, %s clés équipe/ligue/rôle",
            len(records),
            len(rows_by_key),
        )
        return cls(dict(rows_by_key))

    def compute_team_hsh_stats(
        self,
        team_id: int,
        league_id: int,
        as_home: bool = True,
        last_n: int = 20,
        before_date=None,
        exclude_match_id: int = None,
    ) -> dict:
        rows = self.rows_by_key.get((team_id, league_id, as_home), [])
        if not rows:
            return _empty_stats(as_home)

        if before_date is not None:
            cutoff = _coerce_datetime(before_date)
            dates = self.dates_by_key.get((team_id, league_id, as_home), [])
            end_index = bisect_left(dates, cutoff)
            rows = rows[:end_index]

        if exclude_match_id is not None and rows:
            tail = rows[-(last_n + 1):]
            tail = [row for row in tail if row.get("footystats_id") != exclude_match_id]
            rows = tail

        if last_n:
            rows = rows[-last_n:]

        return _aggregate_team_stats(rows, as_home)


def compute_team_hsh_stats(
    team_id: int,
    league_id: int,
    as_home: bool = True,
    last_n: int = 20,
    before_date = None,
    exclude_match_id: int = None,
    stats_cache: HistoricalStatsCache = None,
) -> dict:
    """
    Calcule les stats HSH d'une équipe sur ses N derniers matchs.
    Retourne un dict de features.
    """
    if stats_cache is not None:
        return stats_cache.compute_team_hsh_stats(
            team_id=team_id,
            league_id=league_id,
            as_home=as_home,
            last_n=last_n,
            before_date=before_date,
            exclude_match_id=exclude_match_id,
        )

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        field = "home_id" if as_home else "away_id"
        query = f"""
            SELECT hsh_result, hsh_goals_h1, hsh_goals_h2,
                   ht_home, ht_away, ft_home, ft_away
            FROM matches
            WHERE {field} = %s
              AND league_id = %s
              AND has_ht_data = TRUE
              AND hsh_result IS NOT NULL
        """
        params = [team_id, league_id]

        if before_date is not None:
            query += " AND match_date < %s"
            params.append(before_date)

        if exclude_match_id is not None:
            query += " AND footystats_id <> %s"
            params.append(exclude_match_id)

        query += " ORDER BY match_date DESC LIMIT %s"
        params.append(last_n)

        cur.execute(query, params)

        rows = cur.fetchall()

    return _aggregate_team_stats(rows, as_home)


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


def build_match_features(match: dict, league_group: str, stats_cache: HistoricalStatsCache = None) -> dict:
    """
    Construit le vecteur de features pour un match donné.
    Combine les stats domicile + extérieur + contexte ligue.
    """
    home_id   = match["home_id"]
    away_id   = match["away_id"]
    league_id = match["league_id"]
    as_of_date = _coerce_datetime(match.get("match_date"))
    exclude_match_id = match.get("footystats_id")

    # Stats des 20 derniers matchs à domicile/extérieur
    home_stats = compute_team_hsh_stats(
        home_id, league_id, as_home=True, last_n=20,
        before_date=as_of_date, exclude_match_id=exclude_match_id, stats_cache=stats_cache
    )
    away_stats = compute_team_hsh_stats(
        away_id, league_id, as_home=False, last_n=20,
        before_date=as_of_date, exclude_match_id=exclude_match_id, stats_cache=stats_cache
    )

    # Stats des 10 derniers (forme récente)
    home_recent = compute_team_hsh_stats(
        home_id, league_id, as_home=True, last_n=10,
        before_date=as_of_date, exclude_match_id=exclude_match_id, stats_cache=stats_cache
    )
    away_recent = compute_team_hsh_stats(
        away_id, league_id, as_home=False, last_n=10,
        before_date=as_of_date, exclude_match_id=exclude_match_id, stats_cache=stats_cache
    )

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

    stats_cache = HistoricalStatsCache.from_matches(matches)
    rows = []
    for i, match in enumerate(matches):
        if i % 500 == 0:
            logger.info(f"  Features: {i}/{len(matches)}")

        lg = match.get("league_group", "D")
        feats = build_match_features(dict(match), lg, stats_cache=stats_cache)
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
