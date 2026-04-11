# ============================================================
#  HSH PREDICTOR — Base de données PostgreSQL
# ============================================================

import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from config import DATABASE_URL
import logging

logger = logging.getLogger(__name__)


@contextmanager
def get_conn():
    """Context manager pour connexion PostgreSQL."""
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Crée toutes les tables si elles n'existent pas."""
    with get_conn() as conn:
        cur = conn.cursor()

        # ── Table : ligues ──────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS leagues (
                id              SERIAL PRIMARY KEY,
                footystats_id   INTEGER UNIQUE NOT NULL,
                name            TEXT NOT NULL,
                country         TEXT,
                season          TEXT,
                league_group    TEXT DEFAULT 'D',
                total_matches   INTEGER DEFAULT 0,
                ht_data_pct     FLOAT DEFAULT 0.0,
                is_active       BOOLEAN DEFAULT TRUE,
                created_at      TIMESTAMP DEFAULT NOW(),
                updated_at      TIMESTAMP DEFAULT NOW()
            );
        """)

        # ── Table : équipes ─────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                id              SERIAL PRIMARY KEY,
                footystats_id   INTEGER UNIQUE NOT NULL,
                name            TEXT NOT NULL,
                country         TEXT,
                league_id       INTEGER REFERENCES leagues(footystats_id),
                created_at      TIMESTAMP DEFAULT NOW()
            );
        """)

        # ── Table : matchs ──────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                id              SERIAL PRIMARY KEY,
                footystats_id   INTEGER UNIQUE NOT NULL,
                league_id       INTEGER REFERENCES leagues(footystats_id),
                home_id         INTEGER,
                away_id         INTEGER,
                home_name       TEXT,
                away_name       TEXT,
                match_date      TIMESTAMP,
                season          TEXT,
                status          TEXT DEFAULT 'incomplete',

                -- Scores
                ht_home         INTEGER,
                ht_away         INTEGER,
                ft_home         INTEGER,
                ft_away         INTEGER,

                -- Résultat HSH calculé
                hsh_result      TEXT,   -- 'H1', 'H2', 'EQ'
                hsh_goals_h1    INTEGER,
                hsh_goals_h2    INTEGER,

                -- Stats collectées
                home_ppg        FLOAT,
                away_ppg        FLOAT,
                home_xg         FLOAT,
                away_xg         FLOAT,

                has_ht_data     BOOLEAN DEFAULT FALSE,
                created_at      TIMESTAMP DEFAULT NOW()
            );
        """)

        # ── Table : stats équipes (agrégées) ────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS team_stats (
                id                  SERIAL PRIMARY KEY,
                team_id             INTEGER REFERENCES teams(footystats_id),
                league_id           INTEGER REFERENCES leagues(footystats_id),
                season              TEXT,
                matches_played      INTEGER DEFAULT 0,

                -- Stats HSH globales
                pct_h1_wins         FLOAT DEFAULT 0.0,
                pct_h2_wins         FLOAT DEFAULT 0.0,
                pct_eq              FLOAT DEFAULT 0.0,

                -- Stats domicile
                home_goals_h1_avg   FLOAT DEFAULT 0.0,
                home_goals_h2_avg   FLOAT DEFAULT 0.0,
                home_pct_h1_wins    FLOAT DEFAULT 0.0,
                home_pct_h2_wins    FLOAT DEFAULT 0.0,

                -- Stats extérieur
                away_goals_h1_avg   FLOAT DEFAULT 0.0,
                away_goals_h2_avg   FLOAT DEFAULT 0.0,
                away_pct_h1_wins    FLOAT DEFAULT 0.0,
                away_pct_h2_wins    FLOAT DEFAULT 0.0,

                -- Forme récente (5 derniers matchs)
                recent_h1_wins      INTEGER DEFAULT 0,
                recent_h2_wins      INTEGER DEFAULT 0,
                recent_eq           INTEGER DEFAULT 0,

                updated_at          TIMESTAMP DEFAULT NOW(),
                UNIQUE(team_id, league_id, season)
            );
        """)

        # ── Table : prédictions ─────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              SERIAL PRIMARY KEY,
                match_id        INTEGER REFERENCES matches(footystats_id),
                predicted_at    TIMESTAMP DEFAULT NOW(),

                -- Probabilités ML calibrées
                prob_h1         FLOAT,
                prob_h2         FLOAT,
                prob_eq         FLOAT,

                -- Ajustements LLM
                llm_adj_h1      FLOAT DEFAULT 0.0,
                llm_adj_h2      FLOAT DEFAULT 0.0,
                llm_adj_eq      FLOAT DEFAULT 0.0,
                llm_analysis    TEXT,

                -- Probabilités finales
                final_prob_h1   FLOAT,
                final_prob_h2   FLOAT,
                final_prob_eq   FLOAT,

                -- Recommandation
                recommendation  TEXT,   -- 'H1', 'H2', 'EQ', 'NO_BET'
                confidence      FLOAT,
                is_value_bet    BOOLEAN DEFAULT FALSE,
                ev_score        FLOAT,

                -- Kelly
                kelly_fraction  FLOAT,
                suggested_stake FLOAT,

                -- Résultat (rempli après le match)
                actual_result   TEXT,
                is_correct      BOOLEAN,
                profit_loss     FLOAT,

                league_group    TEXT,
                model_version   TEXT DEFAULT 'v1.0'
            );
        """)

        cur.execute("ALTER TABLE matches ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS raw_prob_h1 FLOAT;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS raw_prob_h2 FLOAT;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS raw_prob_eq FLOAT;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS cal_prob_h1 FLOAT;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS cal_prob_h2 FLOAT;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS cal_prob_eq FLOAT;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS odds_h1 FLOAT;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS odds_h2 FLOAT;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS odds_eq FLOAT;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS recommended_odd FLOAT;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS edge_pct FLOAT;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS confidence_level TEXT;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS publication_status TEXT DEFAULT 'draft';")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS published_at TIMESTAMP;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS explanation TEXT;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS key_factor TEXT;")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS context_input TEXT DEFAULT '';")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS run_source TEXT DEFAULT 'manual';")
        cur.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();")

        # ── Table : backtesting ─────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id              SERIAL PRIMARY KEY,
                run_date        TIMESTAMP DEFAULT NOW(),
                league_group    TEXT,
                total_bets      INTEGER,
                correct_bets    INTEGER,
                accuracy        FLOAT,
                roi             FLOAT,
                total_profit    FLOAT,
                model_version   TEXT
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS scheduler_runs (
                id                  SERIAL PRIMARY KEY,
                job_name            TEXT NOT NULL,
                trigger_source      TEXT,
                started_at          TIMESTAMP DEFAULT NOW(),
                finished_at         TIMESTAMP,
                status              TEXT DEFAULT 'running',
                matches_scanned     INTEGER DEFAULT 0,
                predictions_published INTEGER DEFAULT 0,
                warnings            INTEGER DEFAULT 0,
                error_message       TEXT,
                details             JSONB
            );
        """)

        cur.execute("""
            DELETE FROM predictions older
            USING predictions newer
            WHERE older.match_id = newer.match_id
              AND older.match_id IS NOT NULL
              AND older.id < newer.id;
        """)

        # ── Index pour performances ─────────────────────────
        cur.execute("CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_matches_hsh ON matches(hsh_result);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id);")
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_match_unique "
            "ON predictions(match_id) WHERE match_id IS NOT NULL;"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_publication ON predictions(publication_status, published_at);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_scheduler_runs_started_at ON scheduler_runs(started_at DESC);")

        logger.info("✅ Base de données initialisée avec succès.")


def insert_league(data: dict):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO leagues (footystats_id, name, country, season)
            VALUES (%(id)s, %(name)s, %(country)s, %(season)s)
            ON CONFLICT (footystats_id) DO UPDATE
            SET name=EXCLUDED.name,
                country=EXCLUDED.country,
                season=EXCLUDED.season,
                updated_at=NOW()
        """, data)


def insert_match(data: dict):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO matches (
                footystats_id, league_id, home_id, away_id,
                home_name, away_name, match_date, season, status,
                ht_home, ht_away, ft_home, ft_away,
                hsh_result, hsh_goals_h1, hsh_goals_h2,
                home_ppg, away_ppg, has_ht_data
            ) VALUES (
                %(footystats_id)s, %(league_id)s, %(home_id)s, %(away_id)s,
                %(home_name)s, %(away_name)s, %(match_date)s, %(season)s, %(status)s,
                %(ht_home)s, %(ht_away)s, %(ft_home)s, %(ft_away)s,
                %(hsh_result)s, %(hsh_goals_h1)s, %(hsh_goals_h2)s,
                %(home_ppg)s, %(away_ppg)s, %(has_ht_data)s
            )
            ON CONFLICT (footystats_id) DO UPDATE SET
                league_id=EXCLUDED.league_id,
                home_id=EXCLUDED.home_id,
                away_id=EXCLUDED.away_id,
                home_name=EXCLUDED.home_name,
                away_name=EXCLUDED.away_name,
                match_date=EXCLUDED.match_date,
                season=EXCLUDED.season,
                status=EXCLUDED.status,
                hsh_result=EXCLUDED.hsh_result,
                hsh_goals_h1=EXCLUDED.hsh_goals_h1,
                hsh_goals_h2=EXCLUDED.hsh_goals_h2,
                ht_home=EXCLUDED.ht_home,
                ht_away=EXCLUDED.ht_away,
                ft_home=EXCLUDED.ft_home,
                ft_away=EXCLUDED.ft_away,
                home_ppg=EXCLUDED.home_ppg,
                away_ppg=EXCLUDED.away_ppg,
                has_ht_data=EXCLUDED.has_ht_data,
                updated_at=NOW()
        """, data)


def get_matches_for_training(league_group: str = None, min_matches: int = 50):
    """Récupère les matchs avec données HT pour l'entraînement ML."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        query = """
            SELECT m.*, l.league_group, l.name AS league_name, l.country
            FROM matches m
            JOIN leagues l ON m.league_id = l.footystats_id
            WHERE m.has_ht_data = TRUE
            AND m.hsh_result IS NOT NULL
        """
        params = []
        if league_group:
            query += " AND l.league_group = %s"
            params.append(league_group)
        query += " ORDER BY m.match_date ASC"
        cur.execute(query, params)
        return cur.fetchall()


def save_prediction(data: dict):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions (
                match_id, raw_prob_h1, raw_prob_h2, raw_prob_eq,
                cal_prob_h1, cal_prob_h2, cal_prob_eq,
                prob_h1, prob_h2, prob_eq,
                llm_adj_h1, llm_adj_h2, llm_adj_eq, llm_analysis,
                final_prob_h1, final_prob_h2, final_prob_eq,
                recommendation, confidence, is_value_bet, ev_score,
                kelly_fraction, suggested_stake, league_group,
                odds_h1, odds_h2, odds_eq, recommended_odd,
                edge_pct, confidence_level, publication_status,
                published_at, explanation, key_factor, context_input,
                run_source, actual_result, is_correct, profit_loss
            ) VALUES (
                %(match_id)s, %(raw_prob_h1)s, %(raw_prob_h2)s, %(raw_prob_eq)s,
                %(cal_prob_h1)s, %(cal_prob_h2)s, %(cal_prob_eq)s,
                %(prob_h1)s, %(prob_h2)s, %(prob_eq)s,
                %(llm_adj_h1)s, %(llm_adj_h2)s, %(llm_adj_eq)s, %(llm_analysis)s,
                %(final_prob_h1)s, %(final_prob_h2)s, %(final_prob_eq)s,
                %(recommendation)s, %(confidence)s, %(is_value_bet)s, %(ev_score)s,
                %(kelly_fraction)s, %(suggested_stake)s, %(league_group)s,
                %(odds_h1)s, %(odds_h2)s, %(odds_eq)s, %(recommended_odd)s,
                %(edge_pct)s, %(confidence_level)s, %(publication_status)s,
                %(published_at)s, %(explanation)s, %(key_factor)s, %(context_input)s,
                %(run_source)s, %(actual_result)s, %(is_correct)s, %(profit_loss)s
            )
            ON CONFLICT (match_id) WHERE match_id IS NOT NULL
            DO UPDATE SET
                predicted_at=NOW(),
                raw_prob_h1=EXCLUDED.raw_prob_h1,
                raw_prob_h2=EXCLUDED.raw_prob_h2,
                raw_prob_eq=EXCLUDED.raw_prob_eq,
                cal_prob_h1=EXCLUDED.cal_prob_h1,
                cal_prob_h2=EXCLUDED.cal_prob_h2,
                cal_prob_eq=EXCLUDED.cal_prob_eq,
                prob_h1=EXCLUDED.prob_h1,
                prob_h2=EXCLUDED.prob_h2,
                prob_eq=EXCLUDED.prob_eq,
                llm_adj_h1=EXCLUDED.llm_adj_h1,
                llm_adj_h2=EXCLUDED.llm_adj_h2,
                llm_adj_eq=EXCLUDED.llm_adj_eq,
                llm_analysis=EXCLUDED.llm_analysis,
                final_prob_h1=EXCLUDED.final_prob_h1,
                final_prob_h2=EXCLUDED.final_prob_h2,
                final_prob_eq=EXCLUDED.final_prob_eq,
                recommendation=EXCLUDED.recommendation,
                confidence=EXCLUDED.confidence,
                is_value_bet=EXCLUDED.is_value_bet,
                ev_score=EXCLUDED.ev_score,
                kelly_fraction=EXCLUDED.kelly_fraction,
                suggested_stake=EXCLUDED.suggested_stake,
                league_group=EXCLUDED.league_group,
                odds_h1=EXCLUDED.odds_h1,
                odds_h2=EXCLUDED.odds_h2,
                odds_eq=EXCLUDED.odds_eq,
                recommended_odd=EXCLUDED.recommended_odd,
                edge_pct=EXCLUDED.edge_pct,
                confidence_level=EXCLUDED.confidence_level,
                publication_status=EXCLUDED.publication_status,
                published_at=EXCLUDED.published_at,
                explanation=EXCLUDED.explanation,
                key_factor=EXCLUDED.key_factor,
                context_input=EXCLUDED.context_input,
                run_source=EXCLUDED.run_source,
                actual_result=EXCLUDED.actual_result,
                is_correct=EXCLUDED.is_correct,
                profit_loss=EXCLUDED.profit_loss,
                updated_at=NOW()
        """, data)


def create_scheduler_run(job_name: str, trigger_source: str) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO scheduler_runs (job_name, trigger_source, status)
            VALUES (%s, %s, 'running')
            RETURNING id
            """,
            (job_name, trigger_source),
        )
        return cur.fetchone()[0]


def finish_scheduler_run(
    run_id: int,
    status: str,
    matches_scanned: int = 0,
    predictions_published: int = 0,
    warnings: int = 0,
    error_message: str = None,
    details = None,
):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE scheduler_runs
            SET finished_at = NOW(),
                status = %s,
                matches_scanned = %s,
                predictions_published = %s,
                warnings = %s,
                error_message = %s,
                details = %s
            WHERE id = %s
            """,
            (
                status,
                matches_scanned,
                predictions_published,
                warnings,
                error_message,
                psycopg2.extras.Json(details) if details is not None else None,
                run_id,
            ),
        )
