# ============================================================
#  HSH PREDICTOR — Configuration globale
# ============================================================

import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ── API Keys ─────────────────────────────────────────────────
FOOTYSTATS_API_KEY = os.getenv("FOOTYSTATS_API_KEY", "")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")

# ── Base de données ──────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/hsh_predictor"
)

# ── FootyStats API ───────────────────────────────────────────
FOOTYSTATS_BASE_URL = "https://api.football-data-api.com"
API_DELAY_SECONDS   = 2.1          # 1800 req/h → 1 req/2s (marge sécurité)
MAX_REQUESTS_PER_HOUR = 1800

# ── Cotes fixes HSH ─────────────────────────────────────────
COTES = {
    "H1": 3.10,   # 1ère mi-temps
    "EQ": 3.00,   # Égal
    "H2": 2.10,   # 2ème mi-temps
}

# Seuils de rentabilité (1 / cote)
SEUILS_RENTABILITE = {k: round(1 / v, 4) for k, v in COTES.items()}
# H1: 32.26% | EQ: 33.33% | H2: 47.62%

# ── Modèle ML ────────────────────────────────────────────────
RANDOM_STATE     = 42
TEST_SIZE        = 0.2
MIN_MATCHES      = 50       # Minimum matchs pour entraîner un modèle
KELLY_FRACTION   = 0.25     # Kelly 1/4 (sécurisé)
MIN_DATA_QUALITY = 0.80     # 80% minimum de matchs avec HT score

# ── Clustering ligues ────────────────────────────────────────
# Groupes basés sur le profil offensif/défensif
LEAGUE_GROUPS = {
    "A": "Offensif Européen",   # EPL, Liga, Serie A, Bundesliga, Ligue 1...
    "B": "Défensif Américain",  # Serie A Brésil, MLS, Liga MX...
    "C": "Équilibré Asiatique", # J-League, K-League, CSL...
    "D": "Données insuffisantes"
}

# ── Claude (LLM) ─────────────────────────────────────────────
CLAUDE_MODEL     = "claude-sonnet-4-20250514"
CLAUDE_MAX_TOKENS = 800

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = "hsh_predictor.log"

# ── Automatisation / Publication ────────────────────────────
DEFAULT_BANKROLL = float(os.getenv("DEFAULT_BANKROLL", "1000"))
AUTOMATION_ENABLED = _env_bool("AUTOMATION_ENABLED", True)
AUTOMATION_TRIGGER_ON_START = _env_bool("AUTOMATION_TRIGGER_ON_START", True)
AUTOMATION_INTERVAL_SECONDS = int(os.getenv("AUTOMATION_INTERVAL_SECONDS", "1800"))
AUTOMATION_LOOKAHEAD_DAYS = int(os.getenv("AUTOMATION_LOOKAHEAD_DAYS", "0"))
AUTOMATION_MAX_LEAGUES = int(os.getenv("AUTOMATION_MAX_LEAGUES", "0"))
AUTOMATION_CONTEXT = os.getenv("AUTOMATION_CONTEXT", "")
TRACKED_LEAGUES = [
    item.strip() for item in os.getenv("TRACKED_LEAGUES", "").split(",") if item.strip()
]
