#!/usr/bin/env python3
# ============================================================
#  HSH PREDICTOR — Application Web de Consultation
# ============================================================

from __future__ import annotations

import logging
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any

import psycopg2.extras
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from automation import scheduler_snapshot, start_scheduler, stop_scheduler, trigger_scheduler_run
from config import COTES
from database import get_conn, init_db

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="HSH Predictor Platform", version="3.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value


def _day_bounds(target_day: date = None) -> tuple[datetime, datetime]:
    target_day = target_day or date.today()
    start = datetime.combine(target_day, dt_time.min)
    end = start + timedelta(days=1)
    return start, end


def _query_predictions_for_day(target_day: date = None) -> list[dict[str, Any]]:
    start, end = _day_bounds(target_day)
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT p.match_id, p.predicted_at, p.published_at, p.recommendation,
                   p.confidence, p.confidence_level, p.is_value_bet, p.ev_score,
                   p.edge_pct, p.final_prob_h1, p.final_prob_h2, p.final_prob_eq,
                   p.raw_prob_h1, p.raw_prob_h2, p.raw_prob_eq,
                   p.cal_prob_h1, p.cal_prob_h2, p.cal_prob_eq,
                   p.odds_h1, p.odds_h2, p.odds_eq, p.recommended_odd,
                   p.suggested_stake, p.kelly_fraction, p.explanation, p.key_factor,
                   p.actual_result, p.is_correct, p.profit_loss, p.league_group,
                   p.publication_status, m.home_name, m.away_name, m.match_date,
                   m.status, l.name AS league_name, l.country
            FROM predictions p
            JOIN matches m ON m.footystats_id = p.match_id
            JOIN leagues l ON l.footystats_id = m.league_id
            WHERE p.publication_status = 'published'
              AND m.match_date >= %s
              AND m.match_date < %s
            ORDER BY m.match_date ASC, p.predicted_at DESC
            """
            ,
            (start, end),
        )
        rows = cur.fetchall()

    return [_serialize(dict(row)) for row in rows]


def _query_prediction_detail(match_id: int) -> dict[str, Any] | None:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT p.*, m.home_name, m.away_name, m.match_date, m.status,
                   m.hsh_result AS match_result, l.name AS league_name, l.country
            FROM predictions p
            JOIN matches m ON m.footystats_id = p.match_id
            JOIN leagues l ON l.footystats_id = m.league_id
            WHERE p.match_id = %s
            """,
            (match_id,),
        )
        row = cur.fetchone()
    return _serialize(dict(row)) if row else None


def _query_history(limit: int = 30) -> list[dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT p.match_id, p.predicted_at, p.recommendation, p.is_value_bet,
                   p.ev_score, p.edge_pct, p.suggested_stake, p.actual_result,
                   p.is_correct, p.profit_loss, p.confidence_level,
                   m.home_name, m.away_name, m.match_date, l.name AS league_name
            FROM predictions p
            JOIN matches m ON m.footystats_id = p.match_id
            JOIN leagues l ON l.footystats_id = m.league_id
            WHERE p.actual_result IS NOT NULL
            ORDER BY m.match_date DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [_serialize(dict(row)) for row in cur.fetchall()]


def _dashboard_payload() -> dict[str, Any]:
    start, end = _day_bounds()

    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute(
            """
            SELECT
                COUNT(*) AS total_matches,
                COUNT(*) FILTER (WHERE p.publication_status = 'published') AS published_predictions,
                COUNT(*) FILTER (WHERE p.is_value_bet = TRUE AND p.publication_status = 'published') AS value_bets
            FROM matches m
            LEFT JOIN predictions p ON p.match_id = m.footystats_id
            WHERE m.match_date >= %s
              AND m.match_date < %s
            """,
            (start, end),
        )
        today_stats = dict(cur.fetchone() or {})

        cur.execute(
            """
            SELECT
                COUNT(*) AS total_predictions,
                COUNT(*) FILTER (WHERE is_correct = TRUE) AS correct_predictions,
                COALESCE(SUM(profit_loss), 0.0) AS total_profit,
                COALESCE(
                    SUM(profit_loss) / NULLIF(SUM(suggested_stake), 0) * 100,
                    0.0
                ) AS roi_pct
            FROM predictions
            WHERE actual_result IS NOT NULL
            """
        )
        history_stats = dict(cur.fetchone() or {})

        cur.execute(
            """
            SELECT match_id, home_name, away_name, match_date, recommendation,
                   final_prob_h1, final_prob_h2, final_prob_eq,
                   recommended_odd, confidence_level, is_value_bet
            FROM (
                SELECT p.match_id, m.home_name, m.away_name, m.match_date, p.recommendation,
                       p.final_prob_h1, p.final_prob_h2, p.final_prob_eq,
                       p.recommended_odd, p.confidence_level, p.is_value_bet,
                       ROW_NUMBER() OVER (ORDER BY p.ev_score DESC, p.predicted_at DESC) AS rn
                FROM predictions p
                JOIN matches m ON m.footystats_id = p.match_id
                WHERE p.publication_status = 'published'
                  AND m.match_date >= %s
                  AND m.match_date < %s
            ) ranked
            WHERE rn <= 3
            ORDER BY match_date ASC
            """,
            (start, end),
        )
        featured = [_serialize(dict(row)) for row in cur.fetchall()]

    total_predictions = int(history_stats.get("total_predictions") or 0)
    correct_predictions = int(history_stats.get("correct_predictions") or 0)
    hit_rate = round((correct_predictions / total_predictions) * 100, 1) if total_predictions else 0.0

    return {
        "today": {
            "total_matches": int(today_stats.get("total_matches") or 0),
            "published_predictions": int(today_stats.get("published_predictions") or 0),
            "value_bets": int(today_stats.get("value_bets") or 0),
        },
        "performance": {
            "roi_pct": round(float(history_stats.get("roi_pct") or 0.0), 2),
            "hit_rate_pct": hit_rate,
            "total_profit": round(float(history_stats.get("total_profit") or 0.0), 2),
            "graded_predictions": total_predictions,
        },
        "featured": featured,
        "scheduler": scheduler_snapshot(),
        "fixed_odds": COTES,
    }


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    start_scheduler()


@app.on_event("shutdown")
def on_shutdown() -> None:
    stop_scheduler()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM matches")
            match_count = cur.fetchone()[0]
    except Exception as exc:
        logger.warning("Healthcheck DB KO: %s", exc)
        return {"status": "degraded", "database": False, "match_count": 0}

    return {
        "status": "ok",
        "database": True,
        "match_count": match_count,
        "scheduler": scheduler_snapshot(),
    }


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    return _dashboard_payload()


@app.get("/api/predictions/today")
def predictions_today() -> dict[str, Any]:
    return {"predictions": _query_predictions_for_day()}


@app.get("/api/predictions/history")
def predictions_history(limit: int = 30) -> dict[str, Any]:
    return {"history": _query_history(limit=limit)}


@app.get("/api/predictions/{match_id}")
def prediction_detail(match_id: int) -> dict[str, Any]:
    detail = _query_prediction_detail(match_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Pronostic introuvable.")
    return detail


@app.get("/api/admin/status")
def admin_status() -> dict[str, Any]:
    return scheduler_snapshot()


@app.post("/api/admin/run-now")
def admin_run_now() -> dict[str, Any]:
    return trigger_scheduler_run()
