# ============================================================
#  HSH PREDICTOR — Collecte FootyStats API
# ============================================================

import requests
import time
import logging
from datetime import date, datetime, timedelta
from config import (
    FOOTYSTATS_API_KEY, FOOTYSTATS_BASE_URL,
    API_DELAY_SECONDS, MIN_DATA_QUALITY
)
from database import insert_league, insert_match

logger = logging.getLogger(__name__)


class FootyStatsCollector:
    """Collecteur de données FootyStats avec gestion du rate limit."""

    def __init__(self):
        self.api_key  = FOOTYSTATS_API_KEY
        self.base_url = FOOTYSTATS_BASE_URL
        self.delay    = API_DELAY_SECONDS
        self.session  = requests.Session()
        self._req_count = 0
        self._req_start = time.time()

        if not self.api_key:
            raise RuntimeError("FOOTYSTATS_API_KEY manquante.")

    # ── Requête de base ──────────────────────────────────────
    def _get(self, endpoint: str, params: dict = {}) -> dict:
        """Effectue un appel API avec gestion rate limit."""
        params["key"] = self.api_key
        url = f"{self.base_url}/{endpoint}"

        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            self._req_count += 1

            # Log toutes les 50 requêtes
            if self._req_count % 50 == 0:
                elapsed = (time.time() - self._req_start) / 3600
                rate = self._req_count / max(elapsed, 0.001)
                logger.info(f"📊 {self._req_count} requêtes | ~{rate:.0f}/h (limite: 1800/h)")

            time.sleep(self.delay)
            return resp.json()

        except requests.RequestException as e:
            logger.error(f"❌ Erreur API {endpoint}: {e}")
            time.sleep(5)  # Pause sur erreur
            return {}

    # ── Ligues disponibles ───────────────────────────────────
    def get_leagues(self) -> list:
        """Récupère toutes les ligues disponibles sur le compte."""
        logger.info("📥 Récupération des ligues...")
        data = self._get("league-list", {"chosen_leagues_only": "true"})
        leagues = data.get("data", [])
        logger.info(f"✅ {len(leagues)} ligues trouvées")
        return leagues

    # ── Saisons d'une ligue ──────────────────────────────────
    def get_league_seasons(self, league_id: int) -> list:
        """Récupère les saisons disponibles pour une ligue."""
        data = self._get("league-season", {"league_id": league_id})
        return data.get("data", [])

    # ── Matchs d'une saison ──────────────────────────────────
    def get_league_matches(self, season_id: int, page: int = 1) -> dict:
        """Récupère les matchs d'une saison (paginés)."""
        return self._get("league-matches", {
            "season_id": season_id,
            "page":      page,
            "max_per_page": 300
        })

    @staticmethod
    def get_current_season(league: dict) -> dict | None:
        """Choisit la saison la plus récente pour une ligue."""
        seasons = league.get("season", [])
        valid = [season for season in seasons if season.get("id") and season.get("year") is not None]
        if not valid:
            return None
        return max(valid, key=lambda season: int(season.get("year") or 0))

    # ── Stats d'une équipe ───────────────────────────────────
    def get_team_stats(self, team_id: int, season_id: int) -> dict:
        """Récupère les stats détaillées d'une équipe."""
        return self._get("team", {
            "team_id":   team_id,
            "season_id": season_id
        })

    # ── Détails d'un match ───────────────────────────────────
    def get_match_details(self, match_id: int) -> dict:
        """Récupère les détails complets d'un match."""
        return self._get("match", {"match_id": match_id})

    # ── Calcul résultat HSH ──────────────────────────────────
    @staticmethod
    def compute_hsh(ht_home: int, ht_away: int, ft_home: int, ft_away: int):
        """
        Calcule le résultat HSH à partir des scores.
        Retourne : ('H1'|'H2'|'EQ', goals_h1, goals_h2)
        """
        if None in (ht_home, ht_away, ft_home, ft_away):
            return None, None, None

        goals_h1 = ht_home + ht_away
        goals_h2 = (ft_home - ht_home) + (ft_away - ht_away)

        if goals_h2 < 0:  # Données corrompues
            return None, None, None

        if goals_h1 > goals_h2:
            return "H1", goals_h1, goals_h2
        elif goals_h2 > goals_h1:
            return "H2", goals_h1, goals_h2
        else:
            return "EQ", goals_h1, goals_h2

    # ── Parser un match ──────────────────────────────────────
    def parse_match(self, raw: dict, league_id: int, season: str) -> dict:
        """Transforme un match brut FootyStats en dict structuré."""
        # Scores
        ht_home = (
            raw.get("score_halftime_home")
            if raw.get("score_halftime_home") is not None
            else raw.get("ht_goals_team_a")
        )
        ht_away = (
            raw.get("score_halftime_away")
            if raw.get("score_halftime_away") is not None
            else raw.get("ht_goals_team_b")
        )
        ft_home = raw.get("homeGoalCount")
        ft_away = raw.get("awayGoalCount")

        # Conversion en int si possible
        def safe_int(v):
            try: return int(v)
            except: return None

        ht_home, ht_away = safe_int(ht_home), safe_int(ht_away)
        ft_home, ft_away = safe_int(ft_home), safe_int(ft_away)

        has_ht = all(v is not None for v in [ht_home, ht_away, ft_home, ft_away])
        hsh_result, goals_h1, goals_h2 = self.compute_hsh(ht_home, ht_away, ft_home, ft_away)

        # Date
        try:
            match_date = datetime.fromtimestamp(raw.get("date_unix", 0))
        except:
            match_date = None

        return {
            "footystats_id": raw.get("id"),
            "league_id":     league_id,
            "home_id":       raw.get("homeID"),
            "away_id":       raw.get("awayID"),
            "home_name":     raw.get("home_name", ""),
            "away_name":     raw.get("away_name", ""),
            "match_date":    match_date,
            "season":        season,
            "status":        raw.get("status", "incomplete"),
            "ht_home":       ht_home,
            "ht_away":       ht_away,
            "ft_home":       ft_home,
            "ft_away":       ft_away,
            "hsh_result":    hsh_result,
            "hsh_goals_h1":  goals_h1,
            "hsh_goals_h2":  goals_h2,
            "home_ppg":      raw.get("home_ppg"),
            "away_ppg":      raw.get("away_ppg"),
            "has_ht_data":   has_ht,
        }

    # ── Collecte complète d'une ligue ────────────────────────
    def collect_league(self, league: dict, max_seasons: int = 3) -> dict:
        """
        Collecte tous les matchs d'une ligue sur N saisons.
        Retourne un rapport de qualité des données.
        """
        league_name = league.get("name", "Unknown")
        country     = league.get("country", "")

        logger.info(f"🏆 Collecte : {country} — {league_name}")

        # Récupérer les saisons
        league_id = league.get("id")
        if league_id:
            seasons = self.get_league_seasons(league_id)
        else:
            # La réponse `league-list` récente renvoie directement la liste
            # des compétitions/saisons sous `season`, sans identifiant de ligue parent.
            seasons = league.get("season", [])

        if not seasons:
            logger.warning(f"⚠️  Aucune saison trouvée pour {league_name}")
            return {"league_id": league_id or 0, "status": "no_seasons", "matches": 0}

        seasons = seasons[:max_seasons]  # Limiter aux N dernières saisons
        total_matches = 0
        ht_matches    = 0

        for season in seasons:
            season_id   = season.get("id")
            season_name = season.get("year", str(season_id))
            if not season_id:
                continue

            # Sur l'API actuelle, `competition_id` des matchs correspond au `season_id`.
            # On stocke donc une ligne `leagues` par compétition/saison.
            insert_league({
                "id":      season_id,
                "name":    league_name,
                "country": country,
                "season":  str(season_name)
            })

            page = 1

            while True:
                data  = self.get_league_matches(season_id, page)
                raw_matches = data.get("data", [])

                if not raw_matches:
                    break

                for raw in raw_matches:
                    # Ignorer les matchs futurs ou sans score
                    if raw.get("status") not in ["complete", "completed"]:
                        continue

                    parsed = self.parse_match(raw, season_id, str(season_name))
                    if parsed.get("footystats_id"):
                        insert_match(parsed)
                        total_matches += 1
                        if parsed.get("has_ht_data"):
                            ht_matches += 1

                # Pagination
                pager = data.get("pager", {})
                if page >= pager.get("max_page", 1):
                    break
                page += 1

        ht_quality = ht_matches / max(total_matches, 1)
        status = "ok" if ht_quality >= MIN_DATA_QUALITY else "partial" if ht_quality > 0.3 else "insufficient"

        logger.info(
            f"✅ {league_name} : {total_matches} matchs | "
            f"HT data: {ht_quality:.0%} [{status.upper()}]"
        )

        return {
            "league_id":    league_id or seasons[0].get("id", 0),
            "league_name":  league_name,
            "total_matches": total_matches,
            "ht_matches":   ht_matches,
            "ht_quality":   ht_quality,
            "status":       status
        }

    # ── Collecte complète toutes ligues ─────────────────────
    def collect_all(self, max_seasons: int = 3) -> list:
        """
        Lance la collecte complète sur toutes les ligues du compte.
        Durée estimée : ~15h en arrière-plan.
        """
        leagues = self.get_leagues()
        if not leagues:
            logger.error("❌ Aucune ligue récupérée. Vérifier la clé API.")
            return []

        logger.info(f"🚀 Début collecte — {len(leagues)} ligues | max {max_seasons} saisons")
        logger.info(f"⏱️  Durée estimée : ~{len(leagues) * 3 / 60:.1f} heures")

        reports = []
        for i, league in enumerate(leagues, 1):
            logger.info(f"[{i}/{len(leagues)}]")
            report = self.collect_league(league, max_seasons)
            reports.append(report)

        # Résumé final
        ok      = sum(1 for r in reports if r.get("status") == "ok")
        partial = sum(1 for r in reports if r.get("status") == "partial")
        bad     = sum(1 for r in reports if r.get("status") == "insufficient")
        total   = sum(r.get("total_matches", 0) for r in reports)

        logger.info("=" * 50)
        logger.info(f"🏁 COLLECTE TERMINÉE")
        logger.info(f"   ✅ Ligues OK      : {ok}")
        logger.info(f"   ⚠️  Ligues partial : {partial}")
        logger.info(f"   ❌ Ligues faibles  : {bad}")
        logger.info(f"   📦 Total matchs    : {total:,}")
        logger.info("=" * 50)

        return reports

    # ── Mise à jour quotidienne ──────────────────────────────
    def update_today(self) -> int:
        """Collecte uniquement les nouveaux matchs du jour."""
        report = self.sync_matches_for_date(target_date=date.today(), lookahead_days=0)
        logger.info(f"✅ {report['matches_upserted']} matchs du jour synchronisés")
        return report["matches_upserted"]

    def sync_matches_for_date(
        self,
        target_date: date = None,
        lookahead_days: int = 0,
        max_leagues: int = 0,
        leagues: list[dict] = None,
    ) -> dict:
        """
        Synchronise les matchs d'une fenêtre de dates, y compris les matchs futurs.
        Sert à l'automatisation quotidienne et à la publication des pronostics.
        """
        target_date = target_date or date.today()
        end_date = target_date + timedelta(days=max(lookahead_days, 0))
        leagues = self.get_leagues() if leagues is None else leagues
        selected = leagues[:max_leagues] if max_leagues else leagues

        report = {
            "target_date": target_date.isoformat(),
            "end_date": end_date.isoformat(),
            "leagues_scanned": 0,
            "matches_upserted": 0,
            "completed_matches": 0,
            "upcoming_matches": 0,
            "errors": 0,
        }

        for league in selected:
            season = self.get_current_season(league)
            if not season:
                report["errors"] += 1
                continue

            season_id = season["id"]
            season_name = str(season.get("year"))
            league_name = league.get("name", "Unknown")
            country = league.get("country", "")

            insert_league({
                "id": season_id,
                "name": league_name,
                "country": country,
                "season": season_name,
            })

            report["leagues_scanned"] += 1
            page = 1

            while True:
                data = self.get_league_matches(season_id, page)
                raw_matches = data.get("data", [])
                if not raw_matches:
                    break

                for raw in raw_matches:
                    try:
                        match_date = datetime.fromtimestamp(raw.get("date_unix", 0)).date()
                    except Exception:
                        continue

                    if not (target_date <= match_date <= end_date):
                        continue

                    parsed = self.parse_match(raw, season_id, season_name)
                    if not parsed.get("footystats_id"):
                        continue

                    insert_match(parsed)
                    report["matches_upserted"] += 1
                    if parsed.get("status") in {"complete", "completed"}:
                        report["completed_matches"] += 1
                    else:
                        report["upcoming_matches"] += 1

                pager = data.get("pager", {})
                if page >= pager.get("max_page", 1):
                    break
                page += 1

        logger.info(
            "✅ Sync %s → %s | ligues=%s | matchs=%s | futurs=%s | completes=%s",
            report["target_date"],
            report["end_date"],
            report["leagues_scanned"],
            report["matches_upserted"],
            report["upcoming_matches"],
            report["completed_matches"],
        )
        return report


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    collector = FootyStatsCollector()

    # Test rapide : récupérer les ligues
    leagues = collector.get_leagues()
    print(f"\n📋 {len(leagues)} ligues disponibles sur ton compte FootyStats")
    for l in leagues[:5]:
        print(f"  → {l.get('country')} — {l.get('name')} (ID: {l.get('id')})")
    print("  ...")
