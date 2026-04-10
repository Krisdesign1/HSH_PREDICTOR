# ============================================================
#  HSH PREDICTOR — Analyse LLM (Claude)
# ============================================================

import anthropic
import json
import logging
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS

logger = logging.getLogger(__name__)


def _get_client():
    """Construit le client Anthropic uniquement si une clé est disponible."""
    if not ANTHROPIC_API_KEY:
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


SYSTEM_PROMPT = """Tu es un expert en analyse de matchs de football, spécialisé dans la prédiction du marché "Highest Scoring Half" (HSH) — quelle mi-temps aura le plus de buts.

Les 3 options sont :
- H1 : 1ère mi-temps marque plus
- H2 : 2ème mi-temps marque plus  
- EQ : Égalité entre les deux mi-temps

Cotes fixes du bookmaker :
- H1 : 3.10 (seuil rentabilité : 32.3%)
- EQ : 3.00 (seuil rentabilité : 33.3%)
- H2 : 2.10 (seuil rentabilité : 47.6%)

Ton rôle : analyser le contexte du match et proposer des AJUSTEMENTS de probabilités au modèle ML.
Tu dois retourner UNIQUEMENT un JSON valide, sans texte autour, sans backticks.
"""

ANALYSIS_TEMPLATE = """
Voici les données du match à analyser :

MATCH : {home} vs {away}
LIGUE : {league} ({country})
GROUPE DE LIGUE : {group} — {group_label}

PROBABILITÉS ML (calibrées) :
- P(H1) = {prob_h1:.1%}
- P(H2) = {prob_h2:.1%}
- P(EQ) = {prob_eq:.1%}

STATS HISTORIQUES :
- {home} marque {home_pct_h2:.0%} de ses buts en 2H à domicile
- {away} marque {away_pct_h2:.0%} de ses buts en 2H en extérieur
- Ratio buts 2H/1H combiné : {combined_ratio:.2f}

CONTEXTE FOURNI PAR L'UTILISATEUR :
{context}

Analyse ce match et retourne un JSON avec exactement cette structure :
{{
  "adj_h1": <float entre -0.10 et +0.10>,
  "adj_h2": <float entre -0.10 et +0.10>,
  "adj_eq": <float entre -0.10 et +0.10>,
  "reasoning": "<explication courte en français, max 3 phrases>",
  "confidence": "<LOW|MEDIUM|HIGH>",
  "key_factor": "<le facteur le plus déterminant>"
}}

RÈGLE IMPORTANTE : adj_h1 + adj_h2 + adj_eq doit être proche de 0 (redistribution des probabilités).
"""


def analyze_match(
    match: dict,
    ml_probs: dict,
    league_group: str,
    user_context: str = ""
) -> dict:
    """
    Analyse contextuelle LLM d'un match.

    Args:
        match       : données du match (home_name, away_name, league_id...)
        ml_probs    : probabilités ML calibrées {prob_h1, prob_h2, prob_eq}
        league_group: groupe de ligue ('A', 'B', 'C', 'D')
        user_context: contexte saisi par l'utilisateur (blessures, etc.)

    Returns:
        dict avec ajustements et analyse
    """
    from config import LEAGUE_GROUPS

    client = _get_client()
    if client is None:
        logger.info("ℹ️  ANTHROPIC_API_KEY absente — analyse LLM neutralisée.")
        return _neutral_analysis()

    group_label = LEAGUE_GROUPS.get(league_group, "Inconnu")

    prompt = ANALYSIS_TEMPLATE.format(
        home          = match.get("home_name", "Domicile"),
        away          = match.get("away_name", "Extérieur"),
        league        = match.get("league_name", "Ligue inconnue"),
        country       = match.get("country", ""),
        group         = league_group,
        group_label   = group_label,
        prob_h1       = ml_probs.get("prob_h1", 0.33),
        prob_h2       = ml_probs.get("prob_h2", 0.33),
        prob_eq       = ml_probs.get("prob_eq", 0.33),
        home_pct_h2   = match.get("home_pct_h2", 0.45),
        away_pct_h2   = match.get("away_pct_h2", 0.45),
        combined_ratio= match.get("combined_ratio", 1.0),
        context       = user_context if user_context else "Aucun contexte particulier fourni."
    )

    try:
        response = client.messages.create(
            model      = CLAUDE_MODEL,
            max_tokens = CLAUDE_MAX_TOKENS,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()

        # Nettoyer si backticks présents
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)

        # Valider les clés attendues
        required = ["adj_h1", "adj_h2", "adj_eq", "reasoning", "confidence", "key_factor"]
        for k in required:
            if k not in result:
                result[k] = 0.0 if k.startswith("adj") else "N/A"

        # Normaliser les ajustements (éviter les dérives)
        adj_sum = result["adj_h1"] + result["adj_h2"] + result["adj_eq"]
        if abs(adj_sum) > 0.01:
            correction = adj_sum / 3
            result["adj_h1"] -= correction
            result["adj_h2"] -= correction
            result["adj_eq"] -= correction

        logger.info(f"✅ LLM analyse — {result.get('confidence')} | {result.get('key_factor')}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON invalide du LLM : {e}")
        return _neutral_analysis()
    except Exception as e:
        logger.error(f"❌ Erreur LLM : {e}")
        return _neutral_analysis()


def apply_llm_adjustments(ml_probs: dict, llm_result: dict) -> dict:
    """
    Applique les ajustements LLM aux probabilités ML.
    Garantit que les probabilités finales somment à 1.0
    """
    final_h1 = ml_probs["prob_h1"] + llm_result.get("adj_h1", 0)
    final_h2 = ml_probs["prob_h2"] + llm_result.get("adj_h2", 0)
    final_eq = ml_probs["prob_eq"] + llm_result.get("adj_eq", 0)

    # Clipper entre 0.05 et 0.90
    final_h1 = max(0.05, min(0.90, final_h1))
    final_h2 = max(0.05, min(0.90, final_h2))
    final_eq = max(0.05, min(0.90, final_eq))

    # Renormaliser à 1.0
    total = final_h1 + final_h2 + final_eq
    return {
        "final_prob_h1": round(final_h1 / total, 4),
        "final_prob_h2": round(final_h2 / total, 4),
        "final_prob_eq": round(final_eq / total, 4),
    }


def _neutral_analysis() -> dict:
    """Retourne une analyse neutre en cas d'erreur LLM."""
    return {
        "adj_h1":     0.0,
        "adj_h2":     0.0,
        "adj_eq":     0.0,
        "reasoning":  "Analyse LLM indisponible — probabilités ML utilisées sans ajustement.",
        "confidence": "LOW",
        "key_factor": "N/A"
    }
