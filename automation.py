#!/usr/bin/env python3
# ============================================================
#  HSH PREDICTOR — Automatisation & Publication
# ============================================================

from __future__ import annotations

import logging
import threading
import time
import traceback
from datetime import date, datetime, time as dt_time, timedelta
from typing import Any

import numpy as np
import psycopg2.extras

from collector import FootyStatsCollector
from config import (
    AUTOMATION_CONTEXT,
    AUTOMATION_ENABLED,
    AUTOMATION_INTERVAL_SECONDS,
    AUTOMATION_LOOKAHEAD_DAYS,
    AUTOMATION_MAX_LEAGUES,
    AUTOMATION_TRIGGER_ON_START,
    COTES,
    DEFAULT_BANKROLL,
    LEAGUE_GROUPS,
    TRACKED_LEAGUES,
)
from database import (
    create_scheduler_run,
    finish_scheduler_run,
    get_conn,
    init_db,
    save_prediction,
)
from features import FEATURE_COLUMNS, build_match_features
from llm import analyze_match, apply_llm_adjustments
from model import compute_league_profile, load_model, update_all_league_groups
from value_bet import full_value_analysis

logger = logging.getLogger(__name__)

STATE_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()
WAKE_EVENT = threading.Event()
STOP_EVENT = threading.Event()
THREAD: threading.Thread | None = None

AUTOMATION_STATE: dict[str, Any] = {
    "enabled": AUTOMATION_ENABLED,
    "running": False,
    "last_started_at": None,
    "last_finished_at": None,
    "last_status": "idle",
    "last_error": None,
    "last_report": None,
    "last_run_id": None,
    "last_duration_seconds": None,
}


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value


def _day_bounds(target_day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(target_day, dt_time.min)
    end = start + timedelta(days=1)
    return start, end


def _current_state() -> dict[str, Any]:
    with STATE_LOCK:
        return dict(AUTOMATION_STATE)


def _set_state(**kwargs: Any) -> None:
    with STATE_LOCK:
        AUTOMATION_STATE.update(kwargs)


def _predict_raw_probabilities(features: dict[str, Any], league_group: str) -> dict[str, float]:
    model = load_model(league_group)
    X = np.array([[features.get(col, 0) for col in FEATURE_COLUMNS]])

    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        raw = np.mean(
            [calibrated.estimator.predict_proba(X)[0] for calibrated in model.calibrated_classifiers_],
            axis=0,
        )
    else:
        raw = model.predict_proba(X)[0]

    return {
        "raw_prob_h1": round(float(raw[0]), 4),
        "raw_prob_h2": round(float(raw[1]), 4),
        "raw_prob_eq": round(float(raw[2]), 4),
    }


def _predict_calibrated_probabilities(features: dict[str, Any], league_group: str) -> dict[str, float]:
    model = load_model(league_group)
    X = np.array([[features.get(col, 0) for col in FEATURE_COLUMNS]])
    calibrated = model.predict_proba(X)[0]
    return {
        "prob_h1": round(float(calibrated[0]), 4),
        "prob_h2": round(float(calibrated[1]), 4),
        "prob_eq": round(float(calibrated[2]), 4),
    }


def _load_match(match_id: int) -> dict[str, Any] | None:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT m.*, l.name AS league_name, l.country, l.league_group
            FROM matches m
            JOIN leagues l ON m.league_id = l.footystats_id
            WHERE m.footystats_id = %s
            """,
            (match_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _tracked_leagues(leagues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = leagues
    if TRACKED_LEAGUES:
        names = {name.lower() for name in TRACKED_LEAGUES}
        filtered = [league for league in leagues if league.get("name", "").lower() in names]
    if AUTOMATION_MAX_LEAGUES:
        filtered = filtered[:AUTOMATION_MAX_LEAGUES]
    return filtered


def sync_today_matches(target_day: date = None) -> dict[str, Any]:
    target_day = target_day or date.today()
    collector = FootyStatsCollector()
    leagues = _tracked_leagues(collector.get_leagues())

    report = collector.sync_matches_for_date(
        target_date=target_day,
        lookahead_days=AUTOMATION_LOOKAHEAD_DAYS,
        max_leagues=len(leagues) if leagues else 0,
        leagues=leagues,
    )

    report["tracked_leagues"] = [league.get("name") for league in leagues]
    return report


def _select_publication_candidates(target_day: date = None) -> list[dict[str, Any]]:
    target_day = target_day or date.today()
    day_start, day_end = _day_bounds(target_day)

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT m.*, l.name AS league_name, l.country, l.league_group,
                   p.predicted_at AS last_predicted_at
            FROM matches m
            JOIN leagues l ON l.footystats_id = m.league_id
            LEFT JOIN predictions p ON p.match_id = m.footystats_id
            WHERE m.match_date >= %s
              AND m.match_date < %s
              AND m.match_date >= NOW()
              AND m.status NOT IN ('complete', 'completed')
              AND (
                    p.match_id IS NULL
                 OR p.updated_at IS NULL
                 OR p.updated_at <= NOW() - INTERVAL '30 minutes'
                 OR p.publication_status <> 'published'
              )
            ORDER BY m.match_date ASC
            """,
            (day_start, day_end),
        )
        return [dict(row) for row in cur.fetchall()]


def _settle_completed_predictions() -> dict[str, Any]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT p.match_id, p.recommendation, p.suggested_stake, p.recommended_odd,
                   m.hsh_result
            FROM predictions p
            JOIN matches m ON m.footystats_id = p.match_id
            WHERE m.status IN ('complete', 'completed')
              AND m.hsh_result IS NOT NULL
              AND (p.actual_result IS DISTINCT FROM m.hsh_result OR p.is_correct IS NULL)
            """
        )
        rows = cur.fetchall()

        settled = 0
        for row in rows:
            odd = row["recommended_odd"] or COTES.get(row["recommendation"])
            stake = row["suggested_stake"] or 0.0
            hit = row["recommendation"] == row["hsh_result"] if row["recommendation"] else False
            if row["recommendation"] in COTES and odd and stake:
                profit_loss = round(stake * (odd - 1), 2) if hit else round(-stake, 2)
            else:
                profit_loss = 0.0

            cur.execute(
                """
                UPDATE predictions
                SET actual_result = %s,
                    is_correct = %s,
                    profit_loss = %s,
                    updated_at = NOW()
                WHERE match_id = %s
                """,
                (row["hsh_result"], hit, profit_loss, row["match_id"]),
            )
            settled += 1

    return {"settled_predictions": settled}


def build_prediction_record(match: dict[str, Any], bankroll: float = DEFAULT_BANKROLL, user_context: str = "") -> dict[str, Any]:
    league_group = match.get("league_group") or compute_league_profile(match["league_id"]).get("group", "D")
    features = build_match_features(match, league_group)
    raw_probs = _predict_raw_probabilities(features, league_group)
    calibrated_probs = _predict_calibrated_probabilities(features, league_group)

    llm_match = {
        **match,
        "home_pct_h2": features.get("home_pct_h2", 0.45),
        "away_pct_h2": features.get("away_pct_h2", 0.45),
        "combined_ratio": features.get("combined_ratio", 1.0),
    }

    llm_result = analyze_match(llm_match, calibrated_probs, league_group, user_context)
    final_probs = apply_llm_adjustments(calibrated_probs, llm_result)
    analysis = full_value_analysis(final_probs, bankroll)

    recommendation = analysis["recommendation"]
    recommended_odd = COTES.get(recommendation)
    actual_result = match.get("hsh_result")
    is_correct = None
    profit_loss = None
    if actual_result and recommendation in COTES:
        is_correct = recommendation == actual_result
        stake = analysis.get("kelly", {}).get("stake", 0.0) if analysis.get("kelly") else 0.0
        profit_loss = round(stake * (COTES[recommendation] - 1), 2) if is_correct else round(-stake, 2)

    return {
        "match_id": match["footystats_id"],
        "raw_prob_h1": raw_probs["raw_prob_h1"],
        "raw_prob_h2": raw_probs["raw_prob_h2"],
        "raw_prob_eq": raw_probs["raw_prob_eq"],
        "cal_prob_h1": calibrated_probs["prob_h1"],
        "cal_prob_h2": calibrated_probs["prob_h2"],
        "cal_prob_eq": calibrated_probs["prob_eq"],
        "prob_h1": calibrated_probs["prob_h1"],
        "prob_h2": calibrated_probs["prob_h2"],
        "prob_eq": calibrated_probs["prob_eq"],
        "llm_adj_h1": llm_result.get("adj_h1", 0.0),
        "llm_adj_h2": llm_result.get("adj_h2", 0.0),
        "llm_adj_eq": llm_result.get("adj_eq", 0.0),
        "llm_analysis": llm_result.get("reasoning", ""),
        "final_prob_h1": final_probs["final_prob_h1"],
        "final_prob_h2": final_probs["final_prob_h2"],
        "final_prob_eq": final_probs["final_prob_eq"],
        "recommendation": recommendation,
        "confidence": analysis.get("confidence", 0.0),
        "is_value_bet": analysis.get("is_value_bet", False),
        "ev_score": analysis.get("best_ev", 0.0),
        "kelly_fraction": analysis.get("kelly", {}).get("kelly_fraction", 0.0) if analysis.get("kelly") else 0.0,
        "suggested_stake": analysis.get("kelly", {}).get("stake", 0.0) if analysis.get("kelly") else 0.0,
        "league_group": league_group,
        "odds_h1": COTES["H1"],
        "odds_h2": COTES["H2"],
        "odds_eq": COTES["EQ"],
        "recommended_odd": recommended_odd,
        "edge_pct": analysis.get("edge_pct", 0.0),
        "confidence_level": analysis.get("confidence_level", llm_result.get("confidence", "LOW")),
        "publication_status": "published",
        "published_at": datetime.utcnow(),
        "explanation": llm_result.get("reasoning", ""),
        "key_factor": llm_result.get("key_factor", "N/A"),
        "context_input": user_context,
        "run_source": "automation",
        "actual_result": actual_result,
        "is_correct": is_correct,
        "profit_loss": profit_loss,
        "analysis": _serialize(analysis),
        "raw_probabilities": raw_probs,
        "calibrated_probabilities": calibrated_probs,
        "final_probabilities": final_probs,
        "llm": _serialize(llm_result),
        "feature_summary": {
            "combined_ratio": round(float(features.get("combined_ratio", 0.0)), 4),
            "combined_h2_trend": round(float(features.get("combined_h2_trend", 0.0)), 4),
            "home_recent_pct_h2": round(float(features.get("home_recent_pct_h2", 0.0)), 4),
            "away_recent_pct_h2": round(float(features.get("away_recent_pct_h2", 0.0)), 4),
            "ppg_diff": round(float(features.get("ppg_diff", 0.0)), 4),
        },
        "league_group_label": LEAGUE_GROUPS.get(league_group, "Inconnu"),
    }


def publish_prediction_for_match(match_id: int, bankroll: float = DEFAULT_BANKROLL, user_context: str = "") -> dict[str, Any]:
    match = _load_match(match_id)
    if not match:
        raise ValueError(f"Match {match_id} introuvable.")

    prediction = build_prediction_record(match, bankroll=bankroll, user_context=user_context)
    save_prediction(prediction)
    return prediction


def publish_predictions_for_today(target_day: date = None, bankroll: float = DEFAULT_BANKROLL, user_context: str = "") -> dict[str, Any]:
    target_day = target_day or date.today()
    candidates = _select_publication_candidates(target_day)

    published: list[dict[str, Any]] = []
    warnings = 0
    for match in candidates:
        try:
            published.append(publish_prediction_for_match(match["footystats_id"], bankroll=bankroll, user_context=user_context))
        except Exception as exc:
            warnings += 1
            logger.warning("Publication ignorée pour %s: %s", match.get("footystats_id"), exc)

    return {
        "target_day": target_day.isoformat(),
        "matches_scanned": len(candidates),
        "predictions_published": len(published),
        "warnings": warnings,
        "published_match_ids": [item["match_id"] for item in published],
    }


def run_automation_cycle(trigger_source: str = "scheduler", target_day: date = None) -> dict[str, Any]:
    target_day = target_day or date.today()

    if not RUN_LOCK.acquire(blocking=False):
        return {"status": "skipped", "reason": "cycle_already_running"}

    started = time.time()
    run_id = create_scheduler_run("daily_publication", trigger_source)
    _set_state(
        running=True,
        last_started_at=_utc_now(),
        last_status="running",
        last_error=None,
        last_run_id=run_id,
    )

    try:
        init_db()
        sync_report = sync_today_matches(target_day)
        grouping_report = update_all_league_groups()
        publication_report = publish_predictions_for_today(target_day, bankroll=DEFAULT_BANKROLL, user_context=AUTOMATION_CONTEXT)
        settlement_report = _settle_completed_predictions()

        report = {
            "status": "success",
            "target_day": target_day.isoformat(),
            "sync": sync_report,
            "league_groups": grouping_report,
            "publication": publication_report,
            "settlement": settlement_report,
        }
        finish_scheduler_run(
            run_id,
            status="success",
            matches_scanned=publication_report["matches_scanned"],
            predictions_published=publication_report["predictions_published"],
            warnings=publication_report["warnings"],
            details=report,
        )
        _set_state(
            running=False,
            last_finished_at=_utc_now(),
            last_status="success",
            last_report=_serialize(report),
            last_duration_seconds=round(time.time() - started, 2),
        )
        return report
    except Exception as exc:
        report = {
            "status": "error",
            "target_day": target_day.isoformat(),
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
        logger.exception("Cycle d'automatisation en erreur")
        finish_scheduler_run(
            run_id,
            status="error",
            error_message=str(exc),
            details=report,
        )
        _set_state(
            running=False,
            last_finished_at=_utc_now(),
            last_status="error",
            last_error=str(exc),
            last_report=_serialize(report),
            last_duration_seconds=round(time.time() - started, 2),
        )
        return report
    finally:
        RUN_LOCK.release()


def _scheduler_loop() -> None:
    if AUTOMATION_ENABLED and AUTOMATION_TRIGGER_ON_START:
        WAKE_EVENT.set()

    while not STOP_EVENT.is_set():
        triggered = WAKE_EVENT.wait(timeout=AUTOMATION_INTERVAL_SECONDS)
        if STOP_EVENT.is_set():
            break
        WAKE_EVENT.clear()
        if not AUTOMATION_ENABLED and not triggered:
            continue
        run_automation_cycle(trigger_source="manual" if triggered else "scheduler")


def start_scheduler() -> None:
    global THREAD
    if THREAD and THREAD.is_alive():
        return
    STOP_EVENT.clear()
    WAKE_EVENT.clear()
    THREAD = threading.Thread(target=_scheduler_loop, daemon=True, name="hsh-automation")
    THREAD.start()


def stop_scheduler() -> None:
    STOP_EVENT.set()
    WAKE_EVENT.set()


def trigger_scheduler_run() -> dict[str, Any]:
    WAKE_EVENT.set()
    return {"status": "queued", "queued_at": _utc_now()}


def scheduler_snapshot() -> dict[str, Any]:
    state = _current_state()
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT id, job_name, trigger_source, started_at, finished_at, status,
                   matches_scanned, predictions_published, warnings, error_message
            FROM scheduler_runs
            ORDER BY started_at DESC
            LIMIT 10
            """
        )
        runs = [dict(row) for row in cur.fetchall()]

    return {
        **_serialize(state),
        "interval_seconds": AUTOMATION_INTERVAL_SECONDS,
        "lookahead_days": AUTOMATION_LOOKAHEAD_DAYS,
        "tracked_leagues": TRACKED_LEAGUES,
        "recent_runs": _serialize(runs),
    }
