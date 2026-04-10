#!/usr/bin/env python3
# ============================================================
#  HSH PREDICTOR — Point d'entrée principal
# ============================================================

import argparse
import logging
import os
import sys
from config import LOG_LEVEL, LOG_FILE

# Configuration du logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def cmd_init(args):
    """Initialise la base de données."""
    from database import init_db
    logger.info("🗄️  Initialisation de la base de données...")
    init_db()
    print("✅ Base de données prête.")


def cmd_collect(args):
    """Lance la collecte des données FootyStats."""
    from collector import FootyStatsCollector
    from database import init_db

    init_db()
    collector = FootyStatsCollector()

    if args.today:
        n = collector.update_today()
        print(f"✅ {n} nouveaux matchs collectés.")
    else:
        seasons = args.seasons or 3
        print(f"🚀 Collecte complète — {seasons} saisons par ligue (~15h)")
        print("   (Lance en arrière-plan avec: nohup python main.py collect &)")
        reports = collector.collect_all(max_seasons=seasons)
        ok = sum(1 for r in reports if r.get("status") == "ok")
        print(f"✅ Collecte terminée : {ok}/{len(reports)} ligues avec data complète")


def cmd_cluster(args):
    """Met à jour les groupes de ligues."""
    from model import update_all_league_groups
    logger.info("🔄 Clustering des ligues...")
    counts = update_all_league_groups()
    for g, n in counts.items():
        print(f"  Groupe {g}: {n} ligues")


def cmd_train(args):
    """Entraîne les modèles ML."""
    from model import train_all_models, train_model

    if args.group:
        result = train_model(args.group)
        print(f"✅ Modèle groupe {args.group} : accuracy={result.get('accuracy', 'N/A'):.3f}")
    else:
        results = train_all_models()
        for r in results:
            g = r.get('group')
            a = r.get('accuracy', 0)
            print(f"  Groupe {g} : accuracy={a:.3f}")


def cmd_predict(args):
    """Prédit un ou plusieurs matchs."""
    from predictor import predict_single_match, predict_all_upcoming

    if args.all:
        results = predict_all_upcoming(bankroll=args.bankroll)
        value_bets = [r for r in results if r.get("analysis", {}).get("is_value_bet")]
        print(f"\n💰 {len(value_bets)} value bets sur {len(results)} matchs analysés")
        for r in value_bets:
            m = r["match"]
            a = r["analysis"]
            print(f"  → {m['home_name']} vs {m['away_name']} : "
                  f"{a['recommendation']} | EV={a['best_ev']:.3f} | "
                  f"Mise: {a.get('kelly', {}).get('stake', 0):.0f}$")
    else:
        # Match manuel via CLI
        match = {
            "home_name":    args.home,
            "away_name":    args.away,
            "league_name":  args.league or "Ligue inconnue",
            "league_group": args.group or "A",
            "home_id":      0,
            "away_id":      0,
            "league_id":    0,
            "home_ppg":     1.5,
            "away_ppg":     1.5,
            "home_pct_h2":  0.45,
            "away_pct_h2":  0.45,
            "combined_ratio": 1.0,
            "footystats_id": None,
        }
        predict_single_match(
            match        = match,
            user_context = args.context or "",
            bankroll     = args.bankroll,
            verbose      = True
        )


def cmd_backtest(args):
    """Lance le backtesting."""
    from backtest import run_backtest, run_all_backtests

    if args.all:
        run_all_backtests(days_back=args.days, bankroll=args.bankroll)
    else:
        run_backtest(
            league_group = args.group,
            days_back    = args.days,
            bankroll     = args.bankroll
        )


def cmd_serve(args):
    """Lance l'application web locale."""
    import uvicorn

    host = args.host or os.getenv("HOST", "127.0.0.1")
    port = args.port or int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "webapp:app",
        host=host,
        port=port,
        reload=args.reload,
    )


def main():
    parser = argparse.ArgumentParser(
        description="HSH Predictor — Highest Scoring Half Prediction System"
    )
    sub = parser.add_subparsers(dest="command")

    # ── init ────────────────────────────────────────────────
    sub.add_parser("init", help="Initialiser la base de données")

    # ── collect ─────────────────────────────────────────────
    p_collect = sub.add_parser("collect", help="Collecter les données FootyStats")
    p_collect.add_argument("--today",   action="store_true", help="Seulement les matchs du jour")
    p_collect.add_argument("--seasons", type=int, default=3,  help="Nombre de saisons (défaut: 3)")

    # ── cluster ─────────────────────────────────────────────
    sub.add_parser("cluster", help="Mettre à jour les groupes de ligues")

    # ── train ───────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Entraîner les modèles ML")
    p_train.add_argument("--group", type=str, help="Groupe spécifique (A/B/C)")

    # ── predict ─────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Faire une prédiction")
    p_pred.add_argument("--all",      action="store_true", help="Prédire tous les prochains matchs")
    p_pred.add_argument("--home",     type=str, help="Équipe domicile")
    p_pred.add_argument("--away",     type=str, help="Équipe extérieur")
    p_pred.add_argument("--league",   type=str, help="Nom de la ligue")
    p_pred.add_argument("--group",    type=str, default="A", help="Groupe de ligue (A/B/C)")
    p_pred.add_argument("--context",  type=str, default="", help="Contexte du match")
    p_pred.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll (défaut: 1000$)")

    # ── backtest ─────────────────────────────────────────────
    p_back = sub.add_parser("backtest", help="Lancer le backtesting")
    p_back.add_argument("--all",      action="store_true", help="Tester tous les groupes")
    p_back.add_argument("--group",    type=str, help="Groupe spécifique")
    p_back.add_argument("--days",     type=int, default=90, help="Jours en arrière (défaut: 90)")
    p_back.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll simulée")

    # ── serve ────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="Lancer l'application web")
    p_serve.add_argument("--host", type=str, default=None, help="Host HTTP")
    p_serve.add_argument("--port", type=int, default=None, help="Port HTTP")
    p_serve.add_argument("--reload", action="store_true", help="Activer l'auto-reload")

    args = parser.parse_args()

    commands = {
        "init":     cmd_init,
        "collect":  cmd_collect,
        "cluster":  cmd_cluster,
        "train":    cmd_train,
        "predict":  cmd_predict,
        "backtest": cmd_backtest,
        "serve":    cmd_serve,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        print("""
╔══════════════════════════════════════════╗
║       HSH PREDICTOR — v1.0              ║
║   Highest Scoring Half Prediction       ║
╚══════════════════════════════════════════╝

Commandes disponibles :

  1. Initialiser :
     python main.py init

  2. Collecter les données (première fois ~15h) :
     python main.py collect --seasons 3

  3. Mise à jour quotidienne :
     python main.py collect --today

  4. Clustering des ligues :
     python main.py cluster

  5. Entraîner les modèles :
     python main.py train

  6. Prédire un match :
     python main.py predict --home "Arsenal" --away "Chelsea" \\
       --league "Premier League" --group A \\
       --context "Arsenal sans Saka" --bankroll 500

  7. Prédire tous les prochains matchs :
     python main.py predict --all --bankroll 1000

  8. Backtesting :
     python main.py backtest --all --days 90 --bankroll 1000

Cotes fixes : H1=3.10 | EQ=3.00 | H2=2.10
        """)


if __name__ == "__main__":
    main()
