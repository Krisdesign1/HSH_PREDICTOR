# ============================================================
#  HSH PREDICTOR — Modèle ML (XGBoost + Calibration)
# ============================================================

import numpy as np
import pandas as pd
import joblib
import os
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from config import RANDOM_STATE, TEST_SIZE, MIN_MATCHES, LEAGUE_GROUPS
from features import (
    build_training_dataset, build_match_features,
    FEATURE_COLUMNS, LABEL_COLUMN, LABEL_ENCODER, LABEL_DECODER
)

logger = logging.getLogger(__name__)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


BOOTSTRAP_GROUP_RULES = {
    "A": [
        "premier league", "championship", "league one", "league two",
        "bundesliga", "la liga", "segunda división", "segunda division",
        "serie a", "ligue 1", "ligue 2", "eredivisie", "eerste divisie",
        "liga nos", "ligapro", "pro league", "champions league",
        "europa league", "europa conference league", "coppa italia",
        "nations league", "russian premier league", "liga i", "liga ii",
        "super cup",
    ],
    "B": [
        "mls", "usl championship", "brasil", "brazil", "copa do brasil",
        "argentina", "uruguay", "venezuela", "primera división",
        "primera division",
    ],
    "C": [
        "j1 league", "k league", "a-league", "saudi", "australia",
        "japan", "south korea", "israeli", "liga leumit",
    ],
}

BOOTSTRAP_COUNTRY_GROUPS = {
    "england": "A",
    "germany": "A",
    "spain": "A",
    "italy": "A",
    "france": "A",
    "netherlands": "A",
    "portugal": "A",
    "belgium": "A",
    "romania": "A",
    "russia": "A",
    "usa": "B",
    "brazil": "B",
    "argentina": "B",
    "uruguay": "B",
    "venezuela": "B",
    "japan": "C",
    "south korea": "C",
    "australia": "C",
    "saudi arabia": "C",
    "israel": "C",
}


def model_path_for_group(league_group: str) -> str:
    return os.path.join(MODELS_DIR, f"model_group_{league_group}.joblib")


def has_model_for_group(league_group: str) -> bool:
    return os.path.exists(model_path_for_group(league_group))


def available_model_groups() -> set[str]:
    return {group for group in ("A", "B", "C") if has_model_for_group(group)}


# ── Clustering des ligues ────────────────────────────────────
def assign_league_group(league_stats: dict) -> str:
    """
    Assigne une ligue à un groupe selon son profil.
    league_stats doit contenir : avg_goals, h2_pct, h1_pct
    """
    avg_goals = float(league_stats.get("avg_goals") or 0)
    h2_pct    = float(league_stats.get("h2_pct") or 0)

    if avg_goals >= 2.5 and h2_pct >= 0.42:
        return "A"  # Offensif Européen
    elif avg_goals < 2.0 or h2_pct >= 0.48:
        return "B"  # Défensif / Américain
    elif 2.0 <= avg_goals < 2.5:
        return "C"  # Équilibré Asiatique
    else:
        return "D"  # Insuffisant


def bootstrap_league_group(league_name: str = "", country: str = "") -> str:
    """Assigne un groupe par heuristique quand l'historique local est insuffisant."""
    haystack = f"{country} {league_name}".lower().strip()

    for group, patterns in BOOTSTRAP_GROUP_RULES.items():
        if any(pattern in haystack for pattern in patterns):
            return group

    country_key = (country or "").strip().lower()
    return BOOTSTRAP_COUNTRY_GROUPS.get(country_key, "D")


def compute_league_profile(league_id: int) -> dict:
    """Calcule le profil statistique d'une ligue."""
    from database import get_conn
    import psycopg2.extras

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT
                l.name AS league_name,
                l.country,
                COUNT(*) as total,
                AVG(hsh_goals_h1 + hsh_goals_h2) as avg_goals,
                AVG(CASE WHEN hsh_result = 'H1' THEN 1.0 ELSE 0.0 END) as h1_pct,
                AVG(CASE WHEN hsh_result = 'H2' THEN 1.0 ELSE 0.0 END) as h2_pct,
                AVG(CASE WHEN hsh_result = 'EQ' THEN 1.0 ELSE 0.0 END) as eq_pct
            FROM leagues l
            LEFT JOIN matches m ON m.league_id = l.footystats_id
                AND m.has_ht_data = TRUE
                AND m.hsh_result IS NOT NULL
            WHERE l.footystats_id = %s
            GROUP BY l.name, l.country
        """, (league_id,))
        row = cur.fetchone()

    if not row:
        return {"group": "D", "total": 0}

    total = int(row["total"] or 0)
    fallback_group = bootstrap_league_group(row.get("league_name") or "", row.get("country") or "")
    if total < MIN_MATCHES:
        return {
            "total": total,
            "avg_goals": float(row["avg_goals"] or 0),
            "h1_pct": float(row["h1_pct"] or 0),
            "h2_pct": float(row["h2_pct"] or 0),
            "eq_pct": float(row["eq_pct"] or 0),
            "group": fallback_group,
        }

    return {
        "total":     total,
        "avg_goals": float(row["avg_goals"] or 0),
        "h1_pct":    float(row["h1_pct"] or 0),
        "h2_pct":    float(row["h2_pct"] or 0),
        "eq_pct":    float(row["eq_pct"] or 0),
        "group":     assign_league_group(dict(row))
    }


def update_all_league_groups():
    """Met à jour le groupe de toutes les ligues en DB."""
    from database import get_conn

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT footystats_id FROM leagues WHERE is_active = TRUE")
        leagues = [r[0] for r in cur.fetchall()]

    counts = {g: 0 for g in LEAGUE_GROUPS}
    for lid in leagues:
        profile = compute_league_profile(lid)
        group   = profile.get("group", "D")
        counts[group] += 1
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE leagues SET league_group = %s, updated_at = NOW() WHERE footystats_id = %s",
                (group, lid)
            )

    logger.info(f"✅ Groupes mis à jour : {counts}")
    return counts


# ── Entraînement d'un modèle par groupe ─────────────────────
def train_model(league_group: str) -> dict:
    """
    Entraîne un modèle XGBoost calibré pour un groupe de ligues.
    Retourne les métriques d'évaluation.
    """
    logger.info(f"🧠 Entraînement modèle — Groupe {league_group} ({LEAGUE_GROUPS.get(league_group)})")

    # 1. Charger les données
    df = build_training_dataset(league_group)
    if len(df) < MIN_MATCHES:
        logger.warning(f"⚠️  Données insuffisantes pour groupe {league_group} ({len(df)} matchs)")
        return {"status": "insufficient_data", "group": league_group}

    # 2. Préparer X / y
    X = df[FEATURE_COLUMNS].fillna(0).values
    y = df[LABEL_COLUMN].map(LABEL_ENCODER).values

    logger.info(f"   Dataset: {X.shape[0]} matchs × {X.shape[1]} features")
    logger.info(f"   Classes: H1={sum(y==0)} | H2={sum(y==1)} | EQ={sum(y==2)}")

    # 3. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 4. SMOTE — rééquilibrage des classes
    try:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"   SMOTE appliqué → {X_train.shape[0]} samples")
    except Exception as e:
        logger.warning(f"   SMOTE échoué : {e}")

    # 5. Modèle XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # 6. Calibration isotonique (probabilités fiables)
    calibrated = CalibratedClassifierCV(
        xgb_model,
        method="isotonic",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    )

    calibrated.fit(X_train, y_train)

    # 7. Évaluation
    y_pred  = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logloss  = log_loss(y_test, y_proba)

    logger.info(f"   ✅ Accuracy : {accuracy:.3f}")
    logger.info(f"   ✅ Log Loss : {logloss:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['H1','H2','EQ'])}")

    # 8. Sauvegarder le modèle
    model_path = model_path_for_group(league_group)
    joblib.dump(calibrated, model_path)
    logger.info(f"   💾 Modèle sauvegardé : {model_path}")

    return {
        "status":      "ok",
        "group":       league_group,
        "accuracy":    accuracy,
        "log_loss":    logloss,
        "n_train":     X_train.shape[0],
        "n_test":      X_test.shape[0],
        "model_path":  model_path
    }


def train_all_models() -> list:
    """Entraîne un modèle pour chaque groupe de ligues."""
    results = []
    for group in ["A", "B", "C"]:
        result = train_model(group)
        results.append(result)
    return results


# ── Prédiction ───────────────────────────────────────────────
def load_model(league_group: str, allow_fallback: bool = True):
    """Charge le modèle calibré pour un groupe."""
    model_path = model_path_for_group(league_group)
    if not os.path.exists(model_path):
        if not allow_fallback:
            raise FileNotFoundError(f"Modèle groupe {league_group} introuvable.")
        logger.warning(f"⚠️  Modèle groupe {league_group} non trouvé — fallback groupe A")
        model_path = model_path_for_group("A")
    return joblib.load(model_path)


def predict_match(match: dict, league_group: str) -> dict:
    """
    Prédit les probabilités HSH pour un match.
    Retourne : {prob_h1, prob_h2, prob_eq}
    """
    features = build_match_features(match, league_group)
    X = np.array([[features.get(col, 0) for col in FEATURE_COLUMNS]])

    model  = load_model(league_group)
    probas = model.predict_proba(X)[0]

    # probas[0]=H1, probas[1]=H2, probas[2]=EQ
    return {
        "prob_h1": round(float(probas[0]), 4),
        "prob_h2": round(float(probas[1]), 4),
        "prob_eq": round(float(probas[2]), 4),
    }


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    logger.info("🚀 Démarrage entraînement tous groupes...")
    results = train_all_models()
    for r in results:
        print(f"Groupe {r.get('group')} → Accuracy: {r.get('accuracy', 'N/A'):.3f}")
