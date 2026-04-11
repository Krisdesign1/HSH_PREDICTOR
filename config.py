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


def _env_csv(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


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
MIN_CLASS_COUNT  = int(os.getenv("MIN_CLASS_COUNT", "8"))

# ── Validation temporelle ────────────────────────────────────
TEMPORAL_PROTOCOL_MODE = os.getenv("TEMPORAL_PROTOCOL_MODE", "days").strip().lower()
TEMPORAL_TRAIN_DAYS = int(os.getenv("TEMPORAL_TRAIN_DAYS", "365"))
TEMPORAL_VALID_DAYS = int(os.getenv("TEMPORAL_VALID_DAYS", "60"))
TEMPORAL_TEST_DAYS  = int(os.getenv("TEMPORAL_TEST_DAYS", "30"))
TEMPORAL_STEP_DAYS  = int(os.getenv("TEMPORAL_STEP_DAYS", "30"))
TEMPORAL_TRAIN_MATCHES = int(os.getenv("TEMPORAL_TRAIN_MATCHES", "160"))
TEMPORAL_VALID_MATCHES = int(os.getenv("TEMPORAL_VALID_MATCHES", "40"))
TEMPORAL_TEST_MATCHES  = int(os.getenv("TEMPORAL_TEST_MATCHES", "20"))
TEMPORAL_STEP_MATCHES  = int(os.getenv("TEMPORAL_STEP_MATCHES", "20"))
TEMPORAL_MIN_SUBGROUP_ROWS = int(os.getenv("TEMPORAL_MIN_SUBGROUP_ROWS", "20"))
TEMPORAL_MAX_GAP_DAYS = int(os.getenv("TEMPORAL_MAX_GAP_DAYS", "21"))
TEMPORAL_SEGMENT_MODE = os.getenv("TEMPORAL_SEGMENT_MODE", "season").strip().lower()
TEMPORAL_SEGMENT_SELECTION = os.getenv("TEMPORAL_SEGMENT_SELECTION", "largest").strip().lower()

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
TRACKED_LEAGUES = _env_csv("TRACKED_LEAGUES")
PUBLICATION_ALLOWED_LEAGUES = _env_csv("PUBLICATION_ALLOWED_LEAGUES")
PUBLICATION_ALLOWED_COUNTRIES = _env_csv("PUBLICATION_ALLOWED_COUNTRIES")
