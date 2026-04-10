# ============================================================
#  HSH PREDICTOR — Value Bet & Kelly Criterion
# ============================================================

import logging
from config import COTES, SEUILS_RENTABILITE, KELLY_FRACTION

logger = logging.getLogger(__name__)


def compute_expected_values(final_probs: dict) -> dict:
    """
    Calcule l'Expected Value (EV) pour chaque option HSH.
    EV = P(x) × Cote(x) — si EV > 1.0, c'est un value bet.
    """
    return {
        "ev_h1": round(final_probs["final_prob_h1"] * COTES["H1"], 4),
        "ev_h2": round(final_probs["final_prob_h2"] * COTES["H2"], 4),
        "ev_eq": round(final_probs["final_prob_eq"] * COTES["EQ"], 4),
    }


def detect_value_bets(final_probs: dict, ev: dict) -> dict:
    """
    Détecte les value bets et choisit la meilleure recommandation.

    Retourne :
    {
        recommendation : 'H1' | 'H2' | 'EQ' | 'NO_BET',
        is_value_bet   : bool,
        best_ev        : float,
        confidence     : float,
        details        : dict
    }
    """
    value_bets = {}

    # Vérifier chaque option
    for outcome, ev_key, prob_key in [
        ("H1", "ev_h1", "final_prob_h1"),
        ("H2", "ev_h2", "final_prob_h2"),
        ("EQ", "ev_eq", "final_prob_eq"),
    ]:
        ev_val   = ev[ev_key]
        prob     = final_probs[prob_key]
        seuil    = SEUILS_RENTABILITE[outcome]
        is_value = ev_val > 1.0 and prob > seuil

        value_bets[outcome] = {
            "ev":       ev_val,
            "prob":     prob,
            "seuil":    seuil,
            "is_value": is_value,
            "edge":     round((prob - seuil) * 100, 2),  # Edge en %
        }

    # Trouver le meilleur value bet
    valid = {k: v for k, v in value_bets.items() if v["is_value"]}

    if not valid:
        return {
            "recommendation": "NO_BET",
            "is_value_bet":   False,
            "best_ev":        max(ev.values()),
            "confidence":     0.0,
            "details":        value_bets
        }

    # Sélectionner le value bet avec le plus grand EV
    best_outcome = max(valid, key=lambda k: valid[k]["ev"])
    best         = valid[best_outcome]

    # Niveau de confiance basé sur l'edge
    edge = best["edge"]
    if edge >= 10:
        confidence_level = "HIGH"
        confidence_score = 0.85
    elif edge >= 5:
        confidence_level = "MEDIUM"
        confidence_score = 0.65
    else:
        confidence_level = "LOW"
        confidence_score = 0.45

    logger.info(
        f"💰 Value Bet détecté : {best_outcome} | "
        f"EV={best['ev']:.3f} | Edge={edge:.1f}% | {confidence_level}"
    )

    return {
        "recommendation":   best_outcome,
        "is_value_bet":     True,
        "best_ev":          best["ev"],
        "confidence":       confidence_score,
        "confidence_level": confidence_level,
        "edge_pct":         edge,
        "details":          value_bets
    }


def kelly_criterion(prob: float, cote: float, bankroll: float = 1000.0) -> dict:
    """
    Calcule la mise optimale selon le Critère de Kelly.

    Formule : f = (p × b - q) / b
    où b = cote - 1, p = probabilité, q = 1 - p

    On utilise Kelly fractionnel (1/4) pour la gestion du risque.
    """
    b = cote - 1
    q = 1 - prob

    kelly_full = (prob * b - q) / b

    if kelly_full <= 0:
        return {
            "kelly_full":     0.0,
            "kelly_fraction": 0.0,
            "stake":          0.0,
            "stake_pct":      0.0,
            "recommendation": "Ne pas miser"
        }

    kelly_frac  = kelly_full * KELLY_FRACTION
    stake       = kelly_frac * bankroll
    stake_pct   = kelly_frac * 100
    gain_net    = stake * b
    gain_total  = stake * cote

    return {
        "kelly_full":       round(kelly_full, 4),
        "kelly_fraction":   round(kelly_frac, 4),
        "stake":            round(stake, 2),
        "stake_pct":        round(stake_pct, 2),
        "gain_net":         round(gain_net, 2),
        "gain_total":       round(gain_total, 2),
        "recommendation":   f"Miser {stake:.2f}$ ({stake_pct:.1f}% de la bankroll)"
    }


def full_value_analysis(final_probs: dict, bankroll: float = 1000.0) -> dict:
    """
    Pipeline complet : EV → Value Bet → Kelly.
    C'est la fonction principale à appeler.
    """
    ev       = compute_expected_values(final_probs)
    vb       = detect_value_bets(final_probs, ev)
    rec      = vb["recommendation"]

    if rec == "NO_BET":
        return {
            **vb,
            "ev":    ev,
            "kelly": None,
            "summary": "❌ Aucun value bet détecté sur ce match."
        }

    # Kelly sur le meilleur pari
    prob     = final_probs[f"final_prob_{rec.lower()}"]
    cote     = COTES[rec]
    kelly    = kelly_criterion(prob, cote, bankroll)

    # Résumé lisible
    cote_label = {
        "H1": "1ère Mi-Temps (3.10)",
        "H2": "2ème Mi-Temps (2.10)",
        "EQ": "Égal (3.00)"
    }

    summary = (
        f"✅ VALUE BET : {cote_label[rec]} | "
        f"P={prob:.1%} | EV={vb['best_ev']:.3f} | "
        f"Mise suggérée : {kelly['stake']:.2f}$ sur {bankroll:.0f}$"
    )

    return {
        **vb,
        "ev":     ev,
        "kelly":  kelly,
        "summary": summary
    }


def format_report(match: dict, final_probs: dict, analysis: dict, llm_result: dict) -> str:
    """
    Génère un rapport texte lisible pour un match.
    """
    sep = "─" * 50
    rec = analysis["recommendation"]
    kelly = analysis.get("kelly") or {}

    cote_names = {"H1": "1ère MI-TEMPS", "H2": "2ème MI-TEMPS", "EQ": "ÉGAL", "NO_BET": "AUCUN PARI"}

    lines = [
        f"\n{'═'*50}",
        f"  HSH PREDICTOR — RAPPORT DE PRÉDICTION",
        f"{'═'*50}",
        f"  Match   : {match.get('home_name')} vs {match.get('away_name')}",
        f"  Ligue   : {match.get('league_name', 'N/A')}",
        sep,
        f"  PROBABILITÉS FINALES",
        f"  1ère MI-TEMPS : {final_probs['final_prob_h1']:.1%}  (seuil: 32.3%)",
        f"  2ème MI-TEMPS : {final_probs['final_prob_h2']:.1%}  (seuil: 47.6%)",
        f"  ÉGAL          : {final_probs['final_prob_eq']:.1%}  (seuil: 33.3%)",
        sep,
        f"  ANALYSE CLAUDE",
        f"  {llm_result.get('reasoning', 'N/A')}",
        f"  Facteur clé : {llm_result.get('key_factor', 'N/A')}",
        f"  Confiance   : {llm_result.get('confidence', 'N/A')}",
        sep,
        f"  VALUE BET : {cote_names.get(rec, rec)}",
    ]

    if rec != "NO_BET":
        lines += [
            f"  EV Score  : {analysis['best_ev']:.3f}",
            f"  Edge      : +{analysis.get('edge_pct', 0):.1f}%",
            f"  Mise Kelly: {kelly.get('stake', 0):.2f}$ ({kelly.get('stake_pct', 0):.1f}% bankroll)",
            f"  Gain si ✓ : {kelly.get('gain_net', 0):.2f}$",
        ]

    lines.append(f"{'═'*50}\n")
    return "\n".join(lines)
