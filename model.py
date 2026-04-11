# ============================================================
#  HSH PREDICTOR — Modèle ML (XGBoost + Calibration)
# ============================================================

import numpy as np
import pandas as pd
import joblib
import os
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from config import (
    RANDOM_STATE,
    TEST_SIZE,
    MIN_MATCHES,
    MIN_CLASS_COUNT,
    LEAGUE_GROUPS,
    TEMPORAL_PROTOCOL_MODE,
    TEMPORAL_TRAIN_DAYS,
    TEMPORAL_VALID_DAYS,
    TEMPORAL_TEST_DAYS,
    TEMPORAL_STEP_DAYS,
    TEMPORAL_TRAIN_MATCHES,
    TEMPORAL_VALID_MATCHES,
    TEMPORAL_TEST_MATCHES,
    TEMPORAL_STEP_MATCHES,
    TEMPORAL_MAX_GAP_DAYS,
    TEMPORAL_SEGMENT_MODE,
    TEMPORAL_SEGMENT_SELECTION,
)
from features import (
    build_training_dataset, build_match_features,
    FEATURE_COLUMNS, LABEL_COLUMN, LABEL_ENCODER, HistoricalStatsCache
)

logger = logging.getLogger(__name__)

MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TEMPORAL_FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if col != "league_group_enc"]
TEMPORAL_GLOBAL_MODEL_NAME = "model_temporal_global.joblib"


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


def temporal_model_path(league_group: str | None = None) -> str:
    if league_group:
        return os.path.join(MODELS_DIR, f"model_temporal_group_{league_group}.joblib")
    return os.path.join(MODELS_DIR, TEMPORAL_GLOBAL_MODEL_NAME)


def temporal_report_path(league_group: str | None = None, report_kind: str = "train") -> str:
    suffix = f"group_{league_group.lower()}" if league_group else "global"
    return os.path.join(REPORTS_DIR, f"{report_kind}_temporal_{suffix}.json")


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _window_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0, "start": None, "end": None}
    return {
        "rows": int(len(df)),
        "start": pd.Timestamp(df["match_date"].min()).isoformat(),
        "end": pd.Timestamp(df["match_date"].max()).isoformat(),
    }


def _temporal_coverage(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "rows": 0,
            "distinct_days": 0,
            "start": None,
            "end": None,
            "span_days": 0,
        }

    start = pd.Timestamp(df["match_date"].min())
    end = pd.Timestamp(df["match_date"].max())
    distinct_days = int(df["match_date"].dt.normalize().nunique())
    span_days = int((end.normalize() - start.normalize()).days)
    return {
        "rows": int(len(df)),
        "distinct_days": distinct_days,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "span_days": span_days,
    }


def _temporal_coverage_from_matches(matches: list[dict] | list) -> dict:
    if not matches:
        return {
            "rows": 0,
            "distinct_days": 0,
            "start": None,
            "end": None,
            "span_days": 0,
        }

    timestamps = []
    for match in matches:
        match_dict = dict(match)
        match_date = match_dict.get("match_date")
        if match_date is not None:
            timestamps.append(pd.Timestamp(match_date))

    if not timestamps:
        return {
            "rows": 0,
            "distinct_days": 0,
            "start": None,
            "end": None,
            "span_days": 0,
        }

    series = pd.Series(timestamps)
    start = pd.Timestamp(series.min())
    end = pd.Timestamp(series.max())
    distinct_days = int(series.dt.normalize().nunique())
    span_days = int((end.normalize() - start.normalize()).days)
    return {
        "rows": int(len(series)),
        "distinct_days": distinct_days,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "span_days": span_days,
    }


def _match_dates_frame(matches: list[dict] | list) -> pd.DataFrame:
    dates = []
    for match in matches:
        match_dict = dict(match)
        match_date = match_dict.get("match_date")
        if match_date is not None:
            dates.append(pd.Timestamp(match_date))
    return pd.DataFrame({"match_date": pd.to_datetime(dates)})


def _segment_matches_by_gap(matches: list[dict] | list, max_gap_days: int = TEMPORAL_MAX_GAP_DAYS) -> list[list[dict]]:
    rows = [dict(match) for match in matches if dict(match).get("match_date") is not None]
    if not rows:
        return []

    rows.sort(key=lambda item: pd.Timestamp(item["match_date"]))
    segments: list[list[dict]] = []
    current_segment: list[dict] = []
    previous_ts = None

    for row in rows:
        current_ts = pd.Timestamp(row["match_date"])
        if previous_ts is not None:
            gap_days = (current_ts.normalize() - previous_ts.normalize()).days
            if gap_days > max_gap_days:
                segments.append(current_segment)
                current_segment = []
        current_segment.append(row)
        previous_ts = current_ts

    if current_segment:
        segments.append(current_segment)

    return segments


def _segment_matches_by_season(matches: list[dict] | list) -> list[list[dict]]:
    rows = [dict(match) for match in matches if dict(match).get("match_date") is not None]
    if not rows:
        return []

    rows.sort(key=lambda item: pd.Timestamp(item["match_date"]))
    grouped: dict[str, list[dict]] = {}
    order: list[str] = []

    for row in rows:
        season_key = str(row.get("season") or "").strip()
        if not season_key:
            season_key = f"missing:{pd.Timestamp(row['match_date']).year}"
        if season_key not in grouped:
            grouped[season_key] = []
            order.append(season_key)
        grouped[season_key].append(row)

    return [grouped[key] for key in order]


def _collect_usable_segments(
    segments: list[list[dict]],
    valid_days: int,
    test_days: int,
    step_days: int,
    max_gap_days: int,
    segment_mode: str,
    protocol_mode: str,
    train_matches: int,
    valid_matches: int,
    test_matches: int,
    step_matches: int,
) -> tuple[list[dict], str | None]:
    last_error = None
    candidates = []

    for index, segment in enumerate(segments):
        coverage = _temporal_coverage_from_matches(segment)
        try:
            _ensure_temporal_coverage(
                _match_dates_frame(segment),
                valid_days=valid_days,
                test_days=test_days,
                step_days=step_days,
                protocol_mode=protocol_mode,
                train_matches=train_matches,
                valid_matches=valid_matches,
                test_matches=test_matches,
                step_matches=step_matches,
            )
            season_values = sorted(
                {
                    str(dict(row).get("season") or "").strip()
                    for row in segment
                    if str(dict(row).get("season") or "").strip()
                }
            )
            candidates.append(
                {
                    "segment": segment,
                    "segment_index": index,
                    "segment_count": len(segments),
                    "coverage": coverage,
                    "max_gap_days": max_gap_days,
                    "segment_mode": segment_mode,
                    "season_values": season_values,
                }
            )
        except ValueError as exc:
            last_error = str(exc)

    return candidates, last_error


def _select_latest_usable_segment(
    matches: list[dict] | list,
    valid_days: int,
    test_days: int,
    step_days: int = 0,
    max_gap_days: int = TEMPORAL_MAX_GAP_DAYS,
    protocol_mode: str = TEMPORAL_PROTOCOL_MODE,
    train_matches: int = TEMPORAL_TRAIN_MATCHES,
    valid_matches: int = TEMPORAL_VALID_MATCHES,
    test_matches: int = TEMPORAL_TEST_MATCHES,
    step_matches: int = TEMPORAL_STEP_MATCHES,
    segment_mode: str = TEMPORAL_SEGMENT_MODE,
    selection_strategy: str = TEMPORAL_SEGMENT_SELECTION,
) -> tuple[list[dict], dict]:
    mode = (segment_mode or "season").strip().lower()
    segment_sets: list[tuple[str, list[list[dict]]]] = []
    if mode == "gap":
        segment_sets.append(("gap", _segment_matches_by_gap(matches, max_gap_days=max_gap_days)))
    else:
        segment_sets.append(("season", _segment_matches_by_season(matches)))
        segment_sets.append(("gap", _segment_matches_by_gap(matches, max_gap_days=max_gap_days)))

    last_error = None
    candidates = []
    for current_mode, segments in segment_sets:
        if not segments:
            continue
        candidates, last_error = _collect_usable_segments(
            segments=segments,
            valid_days=valid_days,
            test_days=test_days,
            step_days=step_days,
            max_gap_days=max_gap_days,
            segment_mode=current_mode,
            protocol_mode=protocol_mode,
            train_matches=train_matches,
            valid_matches=valid_matches,
            test_matches=test_matches,
            step_matches=step_matches,
        )
        if candidates:
            break

    if candidates:
        strategy = (selection_strategy or "largest").strip().lower()
        if strategy == "latest":
            winner = max(candidates, key=lambda item: item["segment_index"])
        else:
            winner = max(
                candidates,
                key=lambda item: (
                    item["coverage"]["rows"],
                    item["coverage"]["distinct_days"],
                    item["segment_index"],
                ),
            )
        return winner["segment"], {
            "segment_index": winner["segment_index"],
            "segment_count": winner["segment_count"],
            "coverage": winner["coverage"],
            "max_gap_days": winner["max_gap_days"],
            "segment_mode": winner["segment_mode"],
            "season_values": winner["season_values"],
            "protocol_mode": (protocol_mode or "days").strip().lower(),
            "selection_strategy": strategy,
            "candidate_segments": [
                {
                    "segment_index": item["segment_index"],
                    "coverage": item["coverage"],
                    "segment_mode": item["segment_mode"],
                    "season_values": item["season_values"],
                }
                for item in candidates
            ],
        }

    raise ValueError(last_error or "Aucun segment temporel ne satisfait les contraintes.")


def _ensure_temporal_coverage(
    df: pd.DataFrame,
    valid_days: int,
    test_days: int,
    step_days: int = 0,
    protocol_mode: str = TEMPORAL_PROTOCOL_MODE,
    train_matches: int = TEMPORAL_TRAIN_MATCHES,
    valid_matches: int = TEMPORAL_VALID_MATCHES,
    test_matches: int = TEMPORAL_TEST_MATCHES,
    step_matches: int = TEMPORAL_STEP_MATCHES,
) -> dict:
    coverage = _temporal_coverage(df)
    protocol = (protocol_mode or "days").strip().lower()

    if coverage["distinct_days"] < 3:
        raise ValueError(
            "Couverture temporelle insuffisante: "
            f"{coverage['distinct_days']} jour(x) distinct(s), minimum requis=3."
        )

    if protocol == "matches":
        required_rows = max(train_matches + valid_matches + test_matches, 1)
        if coverage["rows"] < required_rows:
            raise ValueError(
                "Volume temporel insuffisant: "
                f"{coverage['rows']} match(s), requis>={required_rows}."
            )
    else:
        required_span_days = max(valid_days + test_days + step_days, 2)
        if coverage["span_days"] < required_span_days:
            raise ValueError(
                "Amplitude temporelle insuffisante: "
                f"{coverage['span_days']} jour(s), requis>={required_span_days}."
            )

    return coverage


def _label_distribution(y: np.ndarray) -> dict[str, int]:
    counts = np.bincount(y, minlength=3)
    return {"H1": int(counts[0]), "H2": int(counts[1]), "EQ": int(counts[2])}


def _write_report(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _build_xgb_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_temporal_dataset(league_group: str = None) -> pd.DataFrame:
    """
    Construit un dataset temporel avec métadonnées de match.
    Ce dataset sert aux splits chronologiques et au walk-forward.
    """
    from database import get_matches_for_training

    matches = get_matches_for_training(league_group)
    return build_temporal_dataset_from_matches(matches, league_group=league_group)


def build_temporal_dataset_from_matches(matches: list[dict] | list, league_group: str = None) -> pd.DataFrame:
    logger.info("📦 Dataset temporel : %s matchs source (groupe=%s)", len(matches), league_group)

    stats_cache = HistoricalStatsCache.from_matches(matches)
    rows = []
    for i, match in enumerate(matches):
        match_dict = dict(match)
        match_date = match_dict.get("match_date")
        if match_date is None:
            continue

        if i % 500 == 0:
            logger.info("  Temporal features: %s/%s", i, len(matches))

        features = build_match_features(
            match_dict,
            match_dict.get("league_group", "D"),
            stats_cache=stats_cache,
        )
        row = {column: features.get(column, 0.0) for column in TEMPORAL_FEATURE_COLUMNS}
        row.update(
            {
                "label": match_dict["hsh_result"],
                "match_id": match_dict.get("footystats_id"),
                "match_date": pd.Timestamp(match_date),
                "league_group": match_dict.get("league_group"),
                "league_id": match_dict.get("league_id"),
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("match_date").reset_index(drop=True)
    logger.info(
        "✅ Dataset temporel prêt : %s lignes | période %s → %s",
        len(df),
        pd.Timestamp(df["match_date"].min()).date(),
        pd.Timestamp(df["match_date"].max()).date(),
    )
    return df


def split_temporal_dataset(
    df: pd.DataFrame,
    protocol_mode: str = TEMPORAL_PROTOCOL_MODE,
    train_days: int = TEMPORAL_TRAIN_DAYS,
    valid_days: int = TEMPORAL_VALID_DAYS,
    test_days: int = TEMPORAL_TEST_DAYS,
    train_matches: int = TEMPORAL_TRAIN_MATCHES,
    valid_matches: int = TEMPORAL_VALID_MATCHES,
    test_matches: int = TEMPORAL_TEST_MATCHES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Dataset temporel vide.")

    protocol = (protocol_mode or "days").strip().lower()
    if protocol == "matches":
        required_rows = train_matches + valid_matches + test_matches
        if len(df) < required_rows:
            raise ValueError(
                "Fenêtres temporelles insuffisantes pour le split "
                f"(rows={len(df)}, requis>={required_rows})."
            )

        train_end = len(df) - (valid_matches + test_matches)
        valid_end = len(df) - test_matches
        train_start = max(0, train_end - train_matches)

        train_df = df.iloc[train_start:train_end].copy()
        valid_df = df.iloc[train_end:valid_end].copy()
        test_df = df.iloc[valid_end:].copy()
    else:
        max_date = pd.Timestamp(df["match_date"].max())
        test_start = max_date - pd.Timedelta(days=test_days)
        valid_start = test_start - pd.Timedelta(days=valid_days)
        train_start = valid_start - pd.Timedelta(days=train_days)

        train_df = df[(df["match_date"] >= train_start) & (df["match_date"] < valid_start)].copy()
        valid_df = df[(df["match_date"] >= valid_start) & (df["match_date"] < test_start)].copy()
        test_df = df[df["match_date"] >= test_start].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError(
            "Fenêtres temporelles insuffisantes pour le split "
            f"(train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)})."
        )

    return train_df, valid_df, test_df


def _prepare_xy(df: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_columns].fillna(0).values
    y = df[LABEL_COLUMN].map(LABEL_ENCODER).values
    return X, y


def _apply_smote_if_possible(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    class_counts = np.bincount(y_train, minlength=3)
    if np.min(class_counts) < 2:
        return X_train, y_train, False

    try:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled, True
    except Exception as exc:
        logger.warning("   SMOTE temporel ignoré : %s", exc)
        return X_train, y_train, False


def _evaluate_estimator(estimator, df: pd.DataFrame, feature_columns: list[str]) -> dict:
    X_eval, y_eval = _prepare_xy(df, feature_columns)
    proba = estimator.predict_proba(X_eval)
    pred = np.argmax(proba, axis=1)

    labels = [0, 1, 2]
    y_one_hot = np.eye(len(labels))[y_eval]
    brier = float(np.mean(np.sum((proba - y_one_hot) ** 2, axis=1)))

    return {
        "rows": int(len(df)),
        "accuracy": round(float(accuracy_score(y_eval, pred)), 4),
        "log_loss": round(float(log_loss(y_eval, proba, labels=labels)), 4),
        "brier_score": round(brier, 4),
        "label_distribution": _label_distribution(y_eval),
        "classification_report": classification_report(
            y_eval,
            pred,
            labels=labels,
            target_names=["H1", "H2", "EQ"],
            zero_division=0,
            output_dict=True,
        ),
    }


def _fit_temporal_estimator(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[object, dict]:
    X_train, y_train = _prepare_xy(train_df, feature_columns)
    X_valid, y_valid = _prepare_xy(valid_df, feature_columns)

    train_counts = np.bincount(y_train, minlength=3)
    valid_counts = np.bincount(y_valid, minlength=3)
    if np.min(train_counts) < MIN_CLASS_COUNT:
        raise ValueError(f"Classes insuffisantes en train: {_label_distribution(y_train)}")
    if np.min(valid_counts) < 2:
        raise ValueError(f"Classes insuffisantes en validation: {_label_distribution(y_valid)}")

    X_train_fit, y_train_fit, smote_applied = _apply_smote_if_possible(X_train, y_train)

    base_model = _build_xgb_model()
    base_model.fit(X_train_fit, y_train_fit)

    calibration_method = "sigmoid"
    if len(valid_df) >= 150 and np.min(valid_counts) >= MIN_CLASS_COUNT:
        calibration_method = "isotonic"

    estimator = base_model
    calibration_status = "disabled"
    try:
        calibrated = CalibratedClassifierCV(base_model, method=calibration_method, cv="prefit")
        calibrated.fit(X_valid, y_valid)
        estimator = calibrated
        calibration_status = calibration_method
    except Exception as exc:
        logger.warning("   Calibration temporelle indisponible, fallback brut : %s", exc)

    return estimator, {
        "smote_applied": smote_applied,
        "calibration_method": calibration_status,
        "train_distribution": _label_distribution(y_train),
        "valid_distribution": _label_distribution(y_valid),
    }


def train_temporal_model(
    league_group: str = None,
    protocol_mode: str = TEMPORAL_PROTOCOL_MODE,
    train_days: int = TEMPORAL_TRAIN_DAYS,
    valid_days: int = TEMPORAL_VALID_DAYS,
    test_days: int = TEMPORAL_TEST_DAYS,
    train_matches: int = TEMPORAL_TRAIN_MATCHES,
    valid_matches: int = TEMPORAL_VALID_MATCHES,
    test_matches: int = TEMPORAL_TEST_MATCHES,
    segment_mode: str = TEMPORAL_SEGMENT_MODE,
    selection_strategy: str = TEMPORAL_SEGMENT_SELECTION,
) -> dict:
    """
    Entraîne un modèle global avec protocole chronologique strict.
    Ce pipeline ne modifie pas la prod actuelle tant qu'il n'est pas explicitement adopté.
    """
    from database import get_matches_for_training

    scope = f"group_{league_group}" if league_group else "global"
    logger.info("🧠 Entraînement temporel — scope=%s", scope)

    matches = get_matches_for_training(league_group)
    source_rows = len(matches)
    if source_rows < MIN_MATCHES:
        return {"status": "insufficient_data", "scope": scope, "rows": source_rows}

    try:
        usable_matches, segment_meta = _select_latest_usable_segment(
            matches,
            valid_days=valid_days,
            test_days=test_days,
            protocol_mode=protocol_mode,
            train_matches=train_matches,
            valid_matches=valid_matches,
            test_matches=test_matches,
            max_gap_days=TEMPORAL_MAX_GAP_DAYS,
            segment_mode=segment_mode,
            selection_strategy=selection_strategy,
        )
    except ValueError as exc:
        return {
            "status": "insufficient_temporal_span",
            "scope": scope,
            "rows": source_rows,
            "coverage": _temporal_coverage_from_matches(matches),
            "error": str(exc),
        }

    coverage = segment_meta["coverage"]
    df = build_temporal_dataset_from_matches(usable_matches, league_group=league_group)

    train_df, valid_df, test_df = split_temporal_dataset(
        df,
        protocol_mode=protocol_mode,
        train_days=train_days,
        valid_days=valid_days,
        test_days=test_days,
        train_matches=train_matches,
        valid_matches=valid_matches,
        test_matches=test_matches,
    )

    estimator, fit_meta = _fit_temporal_estimator(train_df, valid_df, TEMPORAL_FEATURE_COLUMNS)
    valid_metrics = _evaluate_estimator(estimator, valid_df, TEMPORAL_FEATURE_COLUMNS)
    test_metrics = _evaluate_estimator(estimator, test_df, TEMPORAL_FEATURE_COLUMNS)

    model_path = temporal_model_path(league_group)
    report_path = temporal_report_path(league_group, "train")

    bundle = {
        "model": estimator,
        "feature_columns": list(TEMPORAL_FEATURE_COLUMNS),
        "trained_at": _utc_now(),
        "scope": scope,
        "protocol": {
            "protocol_mode": protocol_mode,
            "train_days": train_days,
            "valid_days": valid_days,
            "test_days": test_days,
            "train_matches": train_matches,
            "valid_matches": valid_matches,
            "test_matches": test_matches,
            "segment_mode": segment_mode,
            "selection_strategy": selection_strategy,
        },
        "segment": segment_meta,
        "coverage": coverage,
        "splits": {
            "train": _window_summary(train_df),
            "valid": _window_summary(valid_df),
            "test": _window_summary(test_df),
        },
        "fit_meta": fit_meta,
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }
    joblib.dump(bundle, model_path)

    report = {
        "status": "ok",
        "scope": scope,
        "trained_at": bundle["trained_at"],
        "model_path": model_path,
        "report_path": report_path,
        "feature_columns": list(TEMPORAL_FEATURE_COLUMNS),
        "protocol": bundle["protocol"],
        "segment": segment_meta,
        "coverage": coverage,
        "splits": bundle["splits"],
        "fit_meta": fit_meta,
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }
    _write_report(report_path, report)
    return report


def _iter_walk_forward_windows(
    df: pd.DataFrame,
    protocol_mode: str = TEMPORAL_PROTOCOL_MODE,
    train_days: int = TEMPORAL_TRAIN_DAYS,
    valid_days: int = TEMPORAL_VALID_DAYS,
    test_days: int = TEMPORAL_TEST_DAYS,
    step_days: int = TEMPORAL_STEP_DAYS,
    train_matches: int = TEMPORAL_TRAIN_MATCHES,
    valid_matches: int = TEMPORAL_VALID_MATCHES,
    test_matches: int = TEMPORAL_TEST_MATCHES,
    step_matches: int = TEMPORAL_STEP_MATCHES,
):
    fold_index = 1
    protocol = (protocol_mode or "days").strip().lower()

    if protocol == "matches":
        test_start_idx = train_matches + valid_matches
        while test_start_idx < len(df):
            train_end = test_start_idx - valid_matches
            train_start = max(0, train_end - train_matches)
            valid_start = train_end
            valid_end = test_start_idx
            test_end = min(len(df), test_start_idx + test_matches)

            train_df = df.iloc[train_start:train_end].copy()
            valid_df = df.iloc[valid_start:valid_end].copy()
            test_df = df.iloc[test_start_idx:test_end].copy()

            yield fold_index, train_df, valid_df, test_df
            fold_index += 1
            test_start_idx += step_matches
        return

    min_date = pd.Timestamp(df["match_date"].min())
    max_date = pd.Timestamp(df["match_date"].max())
    test_start = min_date + pd.Timedelta(days=train_days + valid_days)

    while test_start <= max_date:
        valid_start = test_start - pd.Timedelta(days=valid_days)
        train_start = valid_start - pd.Timedelta(days=train_days)
        test_end = test_start + pd.Timedelta(days=test_days)

        train_df = df[(df["match_date"] >= train_start) & (df["match_date"] < valid_start)].copy()
        valid_df = df[(df["match_date"] >= valid_start) & (df["match_date"] < test_start)].copy()
        test_df = df[(df["match_date"] >= test_start) & (df["match_date"] < test_end)].copy()

        yield fold_index, train_df, valid_df, test_df
        fold_index += 1
        test_start += pd.Timedelta(days=step_days)


def walk_forward_temporal_evaluation(
    league_group: str = None,
    protocol_mode: str = TEMPORAL_PROTOCOL_MODE,
    train_days: int = TEMPORAL_TRAIN_DAYS,
    valid_days: int = TEMPORAL_VALID_DAYS,
    test_days: int = TEMPORAL_TEST_DAYS,
    step_days: int = TEMPORAL_STEP_DAYS,
    train_matches: int = TEMPORAL_TRAIN_MATCHES,
    valid_matches: int = TEMPORAL_VALID_MATCHES,
    test_matches: int = TEMPORAL_TEST_MATCHES,
    step_matches: int = TEMPORAL_STEP_MATCHES,
    segment_mode: str = TEMPORAL_SEGMENT_MODE,
    selection_strategy: str = TEMPORAL_SEGMENT_SELECTION,
) -> dict:
    """
    Évalue le modèle en walk-forward. Chaque fold entraîne sur le passé,
    calibre sur une fenêtre de validation, puis score une fenêtre future.
    """
    from database import get_matches_for_training

    scope = f"group_{league_group}" if league_group else "global"
    matches = get_matches_for_training(league_group)
    source_rows = len(matches)
    if source_rows < MIN_MATCHES:
        return {"status": "insufficient_data", "scope": scope, "rows": source_rows}

    try:
        usable_matches, segment_meta = _select_latest_usable_segment(
            matches,
            valid_days=valid_days,
            test_days=test_days,
            step_days=step_days,
            protocol_mode=protocol_mode,
            train_matches=train_matches,
            valid_matches=valid_matches,
            test_matches=test_matches,
            step_matches=step_matches,
            max_gap_days=TEMPORAL_MAX_GAP_DAYS,
            segment_mode=segment_mode,
            selection_strategy=selection_strategy,
        )
    except ValueError as exc:
        return {
            "status": "insufficient_temporal_span",
            "scope": scope,
            "rows": source_rows,
            "coverage": _temporal_coverage_from_matches(matches),
            "error": str(exc),
        }

    coverage = segment_meta["coverage"]
    df = build_temporal_dataset_from_matches(usable_matches, league_group=league_group)

    folds = []
    predictions = []
    skipped = []

    for fold_idx, train_df, valid_df, test_df in _iter_walk_forward_windows(
        df,
        protocol_mode=protocol_mode,
        train_days=train_days,
        valid_days=valid_days,
        test_days=test_days,
        step_days=step_days,
        train_matches=train_matches,
        valid_matches=valid_matches,
        test_matches=test_matches,
        step_matches=step_matches,
    ):
        if train_df.empty or valid_df.empty or test_df.empty:
            skipped.append({"fold": fold_idx, "reason": "empty_window"})
            continue

        try:
            estimator, fit_meta = _fit_temporal_estimator(train_df, valid_df, TEMPORAL_FEATURE_COLUMNS)
        except Exception as exc:
            skipped.append({"fold": fold_idx, "reason": str(exc)})
            continue

        fold_metrics = _evaluate_estimator(estimator, test_df, TEMPORAL_FEATURE_COLUMNS)
        fold_metrics.update(
            {
                "fold": fold_idx,
                "train": _window_summary(train_df),
                "valid": _window_summary(valid_df),
                "test": _window_summary(test_df),
                "fit_meta": fit_meta,
            }
        )
        folds.append(fold_metrics)

        X_test, y_test = _prepare_xy(test_df, TEMPORAL_FEATURE_COLUMNS)
        proba = estimator.predict_proba(X_test)
        pred = np.argmax(proba, axis=1)
        labels = {0: "H1", 1: "H2", 2: "EQ"}
        for row, probs, y_true, y_pred in zip(test_df.to_dict("records"), proba, y_test, pred):
            predictions.append(
                {
                    "fold": fold_idx,
                    "match_id": row.get("match_id"),
                    "match_date": pd.Timestamp(row["match_date"]).isoformat(),
                    "league_group": row.get("league_group"),
                    "actual_result": labels[int(y_true)],
                    "predicted_result": labels[int(y_pred)],
                    "prob_h1": float(probs[0]),
                    "prob_h2": float(probs[1]),
                    "prob_eq": float(probs[2]),
                }
            )

    if not predictions:
        return {
            "status": "no_evaluable_windows",
            "scope": scope,
            "folds_run": 0,
            "folds_skipped": skipped,
        }

    pred_df = pd.DataFrame(predictions)
    y_true = pred_df["actual_result"].map(LABEL_ENCODER).values
    y_pred = pred_df["predicted_result"].map(LABEL_ENCODER).values
    y_proba = pred_df[["prob_h1", "prob_h2", "prob_eq"]].values
    y_one_hot = np.eye(3)[y_true]

    aggregate = {
        "rows": int(len(pred_df)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "log_loss": round(float(log_loss(y_true, y_proba, labels=[0, 1, 2])), 4),
        "brier_score": round(float(np.mean(np.sum((y_proba - y_one_hot) ** 2, axis=1))), 4),
        "label_distribution": _label_distribution(y_true),
    }

    report = {
        "status": "ok",
        "scope": scope,
        "generated_at": _utc_now(),
        "segment": segment_meta,
        "coverage": coverage,
        "protocol": {
            "protocol_mode": protocol_mode,
            "train_days": train_days,
            "valid_days": valid_days,
            "test_days": test_days,
            "step_days": step_days,
            "train_matches": train_matches,
            "valid_matches": valid_matches,
            "test_matches": test_matches,
            "step_matches": step_matches,
            "segment_mode": segment_mode,
            "selection_strategy": selection_strategy,
        },
        "folds_run": len(folds),
        "folds_skipped": skipped,
        "aggregate_metrics": aggregate,
        "fold_metrics": folds,
        "predictions": predictions,
        "report_path": temporal_report_path(league_group, "walk_forward"),
    }
    _write_report(report["report_path"], report)
    return report


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
