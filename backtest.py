# ============================================================
#  HSH PREDICTOR — Backtesting
# ============================================================

import logging
from datetime import datetime, timedelta
from config import (
    COTES,
    TEMPORAL_MIN_SUBGROUP_ROWS,
    TEMPORAL_PROTOCOL_MODE,
    TEMPORAL_TRAIN_DAYS,
    TEMPORAL_VALID_DAYS,
    TEMPORAL_TEST_DAYS,
    TEMPORAL_STEP_DAYS,
    TEMPORAL_TRAIN_MATCHES,
    TEMPORAL_VALID_MATCHES,
    TEMPORAL_TEST_MATCHES,
    TEMPORAL_STEP_MATCHES,
)
from database import get_conn
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


def run_temporal_backtest(
    league_group: str = None,
    bankroll: float = 1000.0,
    protocol_mode: str = TEMPORAL_PROTOCOL_MODE,
    train_days: int = TEMPORAL_TRAIN_DAYS,
    valid_days: int = TEMPORAL_VALID_DAYS,
    test_days: int = TEMPORAL_TEST_DAYS,
    step_days: int = TEMPORAL_STEP_DAYS,
    train_matches: int = TEMPORAL_TRAIN_MATCHES,
    valid_matches: int = TEMPORAL_VALID_MATCHES,
    test_matches: int = TEMPORAL_TEST_MATCHES,
    step_matches: int = TEMPORAL_STEP_MATCHES,
) -> dict:
    """
    Backtest walk-forward strict.
    Les probabilités proviennent de folds entraînés uniquement sur le passé.
    """
    from model import walk_forward_temporal_evaluation

    evaluation = walk_forward_temporal_evaluation(
        league_group=league_group,
        protocol_mode=protocol_mode,
        train_days=train_days,
        valid_days=valid_days,
        test_days=test_days,
        step_days=step_days,
        train_matches=train_matches,
        valid_matches=valid_matches,
        test_matches=test_matches,
        step_matches=step_matches,
    )
    if evaluation.get("status") != "ok":
        logger.warning("⚠️  Backtest temporel impossible : %s", evaluation)
        return evaluation

    predictions = sorted(evaluation["predictions"], key=lambda item: item["match_date"])

    current_bankroll = bankroll
    total_bets = 0
    correct_bets = 0
    total_profit = 0.0
    bet_results = []

    for prediction in predictions:
        final_probs = {
            "final_prob_h1": prediction["prob_h1"],
            "final_prob_h2": prediction["prob_h2"],
            "final_prob_eq": prediction["prob_eq"],
        }
        analysis = full_value_analysis(final_probs, current_bankroll)
        actual_result = prediction["actual_result"]

        if not analysis["is_value_bet"]:
            bet_results.append(
                {
                    "match_id": prediction["match_id"],
                    "date": prediction["match_date"],
                    "bet": "NO_BET",
                    "actual": actual_result,
                    "profit": 0.0,
                    "bankroll": round(current_bankroll, 2),
                }
            )
            continue

        rec = analysis["recommendation"]
        stake = analysis.get("kelly", {}).get("stake", 0.0) if analysis.get("kelly") else 0.0
        if stake <= 0:
            continue

        won = rec == actual_result
        profit = round(stake * (COTES[rec] - 1), 2) if won else round(-stake, 2)
        current_bankroll += profit
        total_profit += profit
        total_bets += 1
        if won:
            correct_bets += 1

        bet_results.append(
            {
                "match_id": prediction["match_id"],
                "date": prediction["match_date"],
                "bet": rec,
                "actual": actual_result,
                "stake": round(stake, 2),
                "profit": profit,
                "bankroll": round(current_bankroll, 2),
                "won": won,
                "edge_pct": analysis.get("edge_pct", 0.0),
                "ev": analysis.get("best_ev", 0.0),
            }
        )

    accuracy = correct_bets / max(total_bets, 1)
    roi = (total_profit / bankroll) * 100

    report = {
        "status": "ok",
        "mode": "temporal_walk_forward",
        "league_group": league_group,
        "protocol": evaluation["protocol"],
        "folds_run": evaluation["folds_run"],
        "folds_skipped": evaluation.get("folds_skipped", []),
        "model_metrics": evaluation["aggregate_metrics"],
        "betting_simulation": {
            "starting_bankroll": bankroll,
            "final_bankroll": round(current_bankroll, 2),
            "total_profit": round(total_profit, 2),
            "roi": round(roi, 2),
            "total_bets": total_bets,
            "correct_bets": correct_bets,
            "accuracy": round(accuracy, 4),
            "no_bets": len(predictions) - total_bets,
        },
        "results": bet_results,
        "report_path": evaluation.get("report_path"),
    }

    _print_temporal_backtest_report(report)

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO backtest_results
                (league_group, total_bets, correct_bets, accuracy, roi, total_profit, model_version)
                VALUES (%s, %s, %s, %s, %s, %s, 'temporal_walk_forward')
                """,
                (
                    league_group,
                    total_bets,
                    correct_bets,
                    accuracy,
                    roi,
                    total_profit,
                ),
            )
        report["db_persisted"] = True
    except Exception as exc:
        logger.warning("⚠️  Sauvegarde DB du backtest temporel impossible : %s", exc)
        report["db_persisted"] = False
        report["persistence_warning"] = str(exc)

    return report


def run_temporal_subgroup_analysis(
    group_by: str = "league_name",
    league_group: str = None,
    min_rows: int = TEMPORAL_MIN_SUBGROUP_ROWS,
    limit: int = 10,
    protocol_mode: str = TEMPORAL_PROTOCOL_MODE,
    train_days: int = TEMPORAL_TRAIN_DAYS,
    valid_days: int = TEMPORAL_VALID_DAYS,
    test_days: int = TEMPORAL_TEST_DAYS,
    step_days: int = TEMPORAL_STEP_DAYS,
    train_matches: int = TEMPORAL_TRAIN_MATCHES,
    valid_matches: int = TEMPORAL_VALID_MATCHES,
    test_matches: int = TEMPORAL_TEST_MATCHES,
    step_matches: int = TEMPORAL_STEP_MATCHES,
) -> dict:
    from model import walk_forward_temporal_subgroup_evaluation

    if min_rows is None:
        min_rows = TEMPORAL_MIN_SUBGROUP_ROWS

    report = walk_forward_temporal_subgroup_evaluation(
        group_by=group_by,
        league_group=league_group,
        min_rows=min_rows,
        protocol_mode=protocol_mode,
        train_days=train_days,
        valid_days=valid_days,
        test_days=test_days,
        step_days=step_days,
        train_matches=train_matches,
        valid_matches=valid_matches,
        test_matches=test_matches,
        step_matches=step_matches,
    )
    if report.get("status") != "ok":
        logger.warning("⚠️  Analyse temporelle par sous-groupe impossible : %s", report)
        return report

    _print_temporal_subgroup_report(report, limit=limit)
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


def _print_temporal_backtest_report(report: dict):
    """Résumé lisible du backtest walk-forward."""
    metrics = report["model_metrics"]
    sim = report["betting_simulation"]
    protocol = report["protocol"]
    sep = "─" * 50

    if protocol.get("protocol_mode") == "matches":
        windows_label = (
            f"train={protocol['train_matches']}m | "
            f"valid={protocol['valid_matches']}m | "
            f"test={protocol['test_matches']}m | "
            f"step={protocol['step_matches']}m"
        )
    else:
        windows_label = (
            f"train={protocol['train_days']}j | "
            f"valid={protocol['valid_days']}j | "
            f"test={protocol['test_days']}j | "
            f"step={protocol['step_days']}j"
        )

    print(f"\n{'═'*50}")
    print(f"  RAPPORT BACKTEST TEMPOREL — {report['league_group'] or 'GLOBAL'}")
    print(f"{'═'*50}")
    print(f"  Protocole    : {protocol.get('protocol_mode', 'days')}")
    print(f"  Fenêtres     : {windows_label}")
    print(f"  Folds        : {report['folds_run']} | skips={len(report['folds_skipped'])}")
    print(sep)
    print(f"  MODEL")
    print(f"  Lignes scorées : {metrics['rows']}")
    print(f"  Accuracy      : {metrics['accuracy']:.1%}")
    print(f"  Log Loss      : {metrics['log_loss']:.4f}")
    print(f"  Brier         : {metrics['brier_score']:.4f}")
    print(sep)
    print(f"  BETTING SIM")
    print(f"  Paris joués   : {sim['total_bets']}")
    print(f"  Corrects      : {sim['correct_bets']}")
    print(f"  Accuracy      : {sim['accuracy']:.1%}")
    print(f"  ROI           : {sim['roi']:+.2f}%")
    print(f"  Profit net    : {sim['total_profit']:+.2f}$")
    print(f"  Bankroll fin  : {sim['final_bankroll']:.2f}$")
    print(f"{'═'*50}\n")


def _print_temporal_subgroup_report(report: dict, limit: int = 10):
    """Résumé lisible d'une comparaison temporelle par sous-groupe."""
    sep = "─" * 88
    group_by = report["group_by"]
    results = report["results"][: max(limit, 1)]

    print(f"\n{'═'*88}")
    print(f"  COMPARAISON TEMPORELLE PAR {group_by.upper()}")
    print(f"{'═'*88}")
    print(
        f"  Groupes évalués : {report['evaluated_group_count']} / {report['source_group_count']} "
        f"(min_rows={report['min_rows']})"
    )
    print(f"  Rapport         : {report['report_path']}")
    print(sep)
    print("  Sous-groupe".ljust(34) + "Rows".rjust(6) + "  Acc".rjust(8) + "  LogLoss".rjust(11) + "  Brier".rjust(9) + "  Folds".rjust(8))
    print(sep)
    for item in results:
        metrics = item["aggregate_metrics"]
        label = str(item.get(group_by, ""))[:34]
        acc_label = f"{metrics['accuracy']:.1%}"
        logloss_label = f"{metrics['log_loss']:.4f}"
        brier_label = f"{metrics['brier_score']:.4f}"
        print(
            f"  {label.ljust(34)}"
            f"{str(metrics['rows']).rjust(6)}"
            f"{acc_label.rjust(8)}"
            f"{logloss_label.rjust(11)}"
            f"{brier_label.rjust(9)}"
            f"{str(item['folds_run']).rjust(8)}"
        )
    print(f"{'═'*88}\n")


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
