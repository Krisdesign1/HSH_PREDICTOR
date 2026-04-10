# ============================================================
#  HSH PREDICTOR — Pipeline Principal
# ============================================================

import logging
from database import get_conn, save_prediction
from model import predict_match, compute_league_profile, assign_league_group
from llm import analyze_match, apply_llm_adjustments
from value_bet import full_value_analysis, format_report
from config import COTES
import psycopg2.extras

logger = logging.getLogger(__name__)


def get_league_group(league_id: int) -> str:
    """Récupère le groupe d'une ligue depuis la DB."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT league_group FROM leagues WHERE footystats_id = %s",
            (league_id,)
        )
        row = cur.fetchone()
    return row[0] if row else "D"


def get_upcoming_matches(limit: int = 20) -> list:
    """Récupère les prochains matchs sans prédiction."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT m.*, l.name as league_name, l.country, l.league_group
            FROM matches m
            JOIN leagues l ON m.league_id = l.footystats_id
            WHERE m.status = 'incomplete'
              AND m.match_date >= NOW()
              AND m.footystats_id NOT IN (SELECT match_id FROM predictions WHERE match_id IS NOT NULL)
            ORDER BY m.match_date ASC
            LIMIT %s
        """, (limit,))
        return cur.fetchall()


def predict_single_match(
    match_id: int = None,
    match: dict = None,
    user_context: str = "",
    bankroll: float = 1000.0,
    verbose: bool = True
) -> dict:
    """
    Pipeline complet pour un seul match.

    Args:
        match_id     : ID FootyStats du match (charge depuis DB)
        match        : dict match direct (si pas d'ID DB)
        user_context : contexte utilisateur (blessures, météo, etc.)
        bankroll     : bankroll pour calcul Kelly
        verbose      : afficher le rapport textuel

    Returns:
        dict complet avec prédiction, value bet, kelly
    """
    # Charger le match depuis DB si ID fourni
    if match_id and not match:
        with get_conn() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""
                SELECT m.*, l.name as league_name, l.country, l.league_group
                FROM matches m
                JOIN leagues l ON m.league_id = l.footystats_id
                WHERE m.footystats_id = %s
            """, (match_id,))
            match = dict(cur.fetchone() or {})

    if not match:
        logger.error("❌ Match introuvable.")
        return {"error": "Match introuvable"}

    league_group = match.get("league_group") or get_league_group(match.get("league_id"))

    logger.info(f"🔮 Prédiction : {match.get('home_name')} vs {match.get('away_name')}")
    logger.info(f"   Ligue : {match.get('league_name')} | Groupe : {league_group}")

    # ── Étape 1 : ML ────────────────────────────────────────
    ml_probs = predict_match(match, league_group)
    logger.info(f"   ML → H1:{ml_probs['prob_h1']:.1%} H2:{ml_probs['prob_h2']:.1%} EQ:{ml_probs['prob_eq']:.1%}")

    # ── Étape 2 : LLM ───────────────────────────────────────
    llm_result = analyze_match(match, ml_probs, league_group, user_context)

    # ── Étape 3 : Fusion ML + LLM ───────────────────────────
    final_probs = apply_llm_adjustments(ml_probs, llm_result)
    logger.info(
        f"   Final → H1:{final_probs['final_prob_h1']:.1%} "
        f"H2:{final_probs['final_prob_h2']:.1%} "
        f"EQ:{final_probs['final_prob_eq']:.1%}"
    )

    # ── Étape 4 : Value Bet + Kelly ─────────────────────────
    analysis = full_value_analysis(final_probs, bankroll)

    # ── Étape 5 : Sauvegarder en DB ─────────────────────────
    if match.get("footystats_id"):
        recommendation = analysis["recommendation"]
        recommended_odd = COTES.get(recommendation)
        actual_result = match.get("hsh_result")
        is_correct = None
        profit_loss = None
        if actual_result and recommendation in COTES:
            is_correct = recommendation == actual_result
            stake = analysis.get("kelly", {}).get("stake", 0.0) if analysis.get("kelly") else 0.0
            profit_loss = round(stake * (COTES[recommendation] - 1), 2) if is_correct else round(-stake, 2)

        save_prediction({
            "match_id":      match["footystats_id"],
            "raw_prob_h1":   ml_probs["prob_h1"],
            "raw_prob_h2":   ml_probs["prob_h2"],
            "raw_prob_eq":   ml_probs["prob_eq"],
            "cal_prob_h1":   ml_probs["prob_h1"],
            "cal_prob_h2":   ml_probs["prob_h2"],
            "cal_prob_eq":   ml_probs["prob_eq"],
            "prob_h1":       ml_probs["prob_h1"],
            "prob_h2":       ml_probs["prob_h2"],
            "prob_eq":       ml_probs["prob_eq"],
            "llm_adj_h1":    llm_result.get("adj_h1", 0),
            "llm_adj_h2":    llm_result.get("adj_h2", 0),
            "llm_adj_eq":    llm_result.get("adj_eq", 0),
            "llm_analysis":  llm_result.get("reasoning", ""),
            "final_prob_h1": final_probs["final_prob_h1"],
            "final_prob_h2": final_probs["final_prob_h2"],
            "final_prob_eq": final_probs["final_prob_eq"],
            "recommendation": analysis["recommendation"],
            "confidence":    analysis["confidence"],
            "is_value_bet":  analysis["is_value_bet"],
            "ev_score":      analysis["best_ev"],
            "kelly_fraction": analysis.get("kelly", {}).get("kelly_fraction", 0) if analysis.get("kelly") else 0,
            "suggested_stake": analysis.get("kelly", {}).get("stake", 0) if analysis.get("kelly") else 0,
            "league_group":  league_group,
            "odds_h1":       COTES["H1"],
            "odds_h2":       COTES["H2"],
            "odds_eq":       COTES["EQ"],
            "recommended_odd": recommended_odd,
            "edge_pct":      analysis.get("edge_pct", 0.0),
            "confidence_level": analysis.get("confidence_level", llm_result.get("confidence", "LOW")),
            "publication_status": "draft",
            "published_at":  None,
            "explanation":   llm_result.get("reasoning", ""),
            "key_factor":    llm_result.get("key_factor", "N/A"),
            "context_input": user_context,
            "run_source":    "manual",
            "actual_result": actual_result,
            "is_correct":    is_correct,
            "profit_loss":   profit_loss,
        })

    # ── Rapport textuel ─────────────────────────────────────
    if verbose:
        report = format_report(match, final_probs, analysis, llm_result)
        print(report)

    return {
        "match":        match,
        "ml_probs":     ml_probs,
        "llm_result":   llm_result,
        "final_probs":  final_probs,
        "analysis":     analysis,
        "league_group": league_group,
    }


def predict_all_upcoming(bankroll: float = 1000.0) -> list:
    """Prédit tous les prochains matchs disponibles."""
    matches = get_upcoming_matches(limit=50)
    logger.info(f"🔮 {len(matches)} matchs à prédire")

    results = []
    for match in matches:
        result = predict_single_match(match=dict(match), bankroll=bankroll, verbose=False)
        results.append(result)
        if result.get("analysis", {}).get("is_value_bet"):
            logger.info(f"  💰 VALUE BET → {match['home_name']} vs {match['away_name']}: "
                       f"{result['analysis']['recommendation']}")

    value_bets = [r for r in results if r.get("analysis", {}).get("is_value_bet")]
    logger.info(f"✅ {len(value_bets)} value bets sur {len(results)} matchs analysés")
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(message)s"
    )

    # Exemple : prédiction manuelle sans DB
    test_match = {
        "footystats_id": None,
        "home_name":     "Arsenal",
        "away_name":     "Chelsea",
        "league_name":   "Premier League",
        "country":       "England",
        "league_id":     None,
        "league_group":  "A",
        "home_id":       1,
        "away_id":       2,
        "home_ppg":      2.1,
        "away_ppg":      1.8,
        "home_pct_h2":   0.48,
        "away_pct_h2":   0.44,
        "combined_ratio": 1.15,
    }

    result = predict_single_match(
        match        = test_match,
        user_context = "Arsenal sans Saka, match crucial pour le titre.",
        bankroll     = 500.0,
        verbose      = True
    )
