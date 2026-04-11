import unittest
from unittest.mock import patch

from automation import _match_in_publication_scope, _publication_scope_snapshot


class PublicationScopeTest(unittest.TestCase):
    def test_publication_scope_accepts_match_inside_allowed_league(self):
        match = {"league_name": "Germany Bundesliga", "country": "Germany"}
        with patch("automation.PUBLICATION_ALLOWED_LEAGUES", ["Germany Bundesliga"]), patch(
            "automation.PUBLICATION_ALLOWED_COUNTRIES", []
        ):
            self.assertTrue(_match_in_publication_scope(match))
            self.assertEqual(
                _publication_scope_snapshot(),
                {"league_names": ["germany bundesliga"], "countries": []},
            )

    def test_publication_scope_rejects_match_outside_allowed_league(self):
        match = {"league_name": "England Premier League", "country": "England"}
        with patch("automation.PUBLICATION_ALLOWED_LEAGUES", ["Germany Bundesliga"]), patch(
            "automation.PUBLICATION_ALLOWED_COUNTRIES", []
        ):
            self.assertFalse(_match_in_publication_scope(match))

    def test_publication_scope_combines_league_and_country_filters(self):
        with patch("automation.PUBLICATION_ALLOWED_LEAGUES", ["Germany Bundesliga"]), patch(
            "automation.PUBLICATION_ALLOWED_COUNTRIES", ["Germany"]
        ):
            self.assertTrue(
                _match_in_publication_scope({"league_name": "Germany Bundesliga", "country": "Germany"})
            )
            self.assertFalse(
                _match_in_publication_scope({"league_name": "Germany Bundesliga", "country": "England"})
            )


if __name__ == "__main__":
    unittest.main()
