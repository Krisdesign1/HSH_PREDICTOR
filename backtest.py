# ============================================================
#  HSH PREDICTOR — Backtesting
# ============================================================

import logging
import pandas as pd
from datetime import datetime, timedelta
from config import COTES, KELLY_FRACTION
from database import get_conn, save_prediction
from model import predict_match
from value_bet import full_value_analysis, kelly_criterion
import psycopg2.extras

logger = logging.getLogger(__name__)


def run_backtest(
    league_group: str = None,
    days_back: int = 90,
    bankroll: float = 1000.0,
    min_ev: float = 1.0
) -> dict:
    """
    Backtest sur les N derniers jours de matchs complétés.

    Args:
        league_group : groupe à tester (None = tous)
        days_back    : nombre de jours en arrière
        bankroll     : bankroll de départ simulée
        min_ev       : seuil EV minimum pour parier

    Returns:
        dict avec métriques ROI, accuracy, profit
    """
    cutoff = datetime.now() - timedelta(days=days_back)

    # Charger les matchs historiques avec résultats connus
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        query = """
            SELECT m.*, l.name as league_name, l.country, l.league_group
            FROM matches m
            JOIN leagues l ON m.league_id = l.footystats_id
            WHERE m.has_ht_data = TRUE
              AND m.hsh_result IS NOT NULL
              AND m.match_date >= %s
              AND m.status IN ('complete', 'completed')
        """
        params = [cutoff]
        if league_group:
            query += " AND l.league_group = %s"
            params.append(league_group)
        query += " ORDER BY m.match_date ASC"
        cur.execute(query, params)
        matches = cur.fetchall()

    if not matches:
        logger.warning(f"⚠️  Aucun match pour le backtest (groupe={league_group}, {days_back}j)")
        return {"status": "no_data"}

    logger.info(f"🔄 Backtest : {len(matches)} matchs | groupe={league_group} | {days_back} jours")

    # Simulation
    current_bankroll = bankroll
    total_bets   = 0
    correct_bets = 0
    total_profit = 0.0
    results      = []

    for match in matches:
        match_dict   = dict(match)
        group        = match_dict.get("league_group", "D")
        actual_result = match_dict["hsh_result"]

        # Prédiction ML (sans LLM pour rapidité du backtest)
        try:
            ml_probs = predict_match(match_dict, group)
        except Exception as e:
            logger.warning(f"⚠️  Skip match {match_dict.get('footystats_id')} : {e}")
            continue

        # On utilise directement les probs ML (pas de LLM en backtest)
        final_probs = {
            "final_prob_h1": ml_probs["prob_h1"],
            "final_prob_h2": ml_probs["prob_h2"],
            "final_prob_eq": ml_probs["prob_eq"],
        }

        analysis = full_value_analysis(final_probs, current_bankroll)

        if not analysis["is_value_bet"]:
            results.append({
                "match": f"{match_dict['home_name']} vs {match_dict['away_name']}",
                "bet":   "NO_BET",
                "actual": actual_result,
                "profit": 0.0,
                "bankroll": current_bankroll,
            })
            continue

        # Calculer la mise
        rec   = analysis["recommendation"]
        prob  = final_probs[f"final_prob_{rec.lower()}"]
        cote  = COTES[rec]
        kelly = kelly_criterion(prob, cote, current_bankroll)
        stake = kelly["stake"]

        if stake <= 0:
            continue

        # Calculer le profit/perte
        won    = (rec == actual_result)
        profit = (stake * cote - stake) if won else -stake
        current_bankroll += profit
        total_profit     += profit
        total_bets       += 1
        if won:
            correct_bets += 1

        results.append({
            "match":     f"{match_dict['home_name']} vs {match_dict['away_name']}",
            "date":      match_dict.get("match_date"),
            "bet":       rec,
            "actual":    actual_result,
            "cote":      cote,
            "stake":     stake,
            "profit":    round(profit, 2),
            "bankroll":  round(current_bankroll, 2),
            "won":       won,
        })

    # Métriques finales
    accuracy = correct_bets / max(total_bets, 1)
    roi      = (total_profit / bankroll) * 100
    no_bets  = len(matches) - total_bets

    report = {
        "status":        "ok",
        "league_group":  league_group,
        "days_back":     days_back,
        "total_matches": len(matches),
        "total_bets":    total_bets,
        "no_bets":       no_bets,
        "correct_bets":  correct_bets,
        "accuracy":      round(accuracy, 4),
        "roi":           round(roi, 2),
        "total_profit":  round(total_profit, 2),
        "final_bankroll": round(current_bankroll, 2),
        "results":       results,
    }

    # Afficher le résumé
    _print_backtest_report(report)

    # Sauvegarder en DB
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO backtest_results
            (league_group, total_bets, correct_bets, accuracy, roi, total_profit, model_version)
            VALUES (%s, %s, %s, %s, %s, %s, 'v1.0')
        """, (league_group, total_bets, correct_bets, accuracy, roi, total_profit))

    return report


def _print_backtest_report(report: dict):
    """Affiche un résumé lisible du backtest."""
    sep = "─" * 50
    roi_emoji = "✅" if report["roi"] >= 0 else "❌"

    print(f"\n{'═'*50}")
    print(f"  RAPPORT BACKTEST — Groupe {report['league_group']}")
    print(f"{'═'*50}")
    print(f"  Période      : {report['days_back']} derniers jours")
    print(f"  Matchs total : {report['total_matches']}")
    print(sep)
    print(f"  Paris joués  : {report['total_bets']}")
    print(f"  Corrects     : {report['correct_bets']}")
    print(f"  Accuracy     : {report['accuracy']:.1%}")
    print(sep)
    print(f"  ROI          : {roi_emoji} {report['roi']:+.1f}%")
    print(f"  Profit net   : {report['total_profit']:+.2f}$")
    print(f"  Bankroll fin : {report['final_bankroll']:.2f}$")
    print(f"{'═'*50}\n")


def run_all_backtests(days_back: int = 90, bankroll: float = 1000.0) -> dict:
    """Lance le backtest sur tous les groupes."""
    all_results = {}
    for group in ["A", "B", "C", None]:
        label = group or "ALL"
        logger.info(f"⚙️  Backtest groupe {label}...")
        result = run_backtest(group, days_back, bankroll)
        all_results[label] = result
    return all_results


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    run_all_backtests(days_back=90, bankroll=1000.0)
