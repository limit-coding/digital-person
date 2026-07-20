import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from digital_mirror.historical_catalog import catalog_root
from digital_mirror.webapp import Settings, create_app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        empty_episodes = Path(self.temporary.name) / "episodes"
        empty_episodes.mkdir()
        settings = Settings(
            episode_root=empty_episodes,
            historical_root=catalog_root(),
            schema_path=Path("schemas/replay_prediction.schema.json"),
            access_password="test-password",
            session_secret="test-session-secret",
            cookie_secure=True,
        )
        self.client = TestClient(create_app(settings=settings), base_url="https://testserver")

    def tearDown(self):
        self.client.close()
        self.temporary.cleanup()

    def test_public_catalog_does_not_require_private_login(self):
        response = self.client.get("/api/public/figures")
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(len(response.json()["figures"]), 11)
        self.assertIn("public", response.headers["cache-control"])

        detail = self.client.get("/api/public/figures/nelson-mandela")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["figure_id"], "nelson-mandela")

    def test_private_events_stay_behind_session_boundary(self):
        response = self.client.get("/api/events")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["cache-control"], "no-store")

        login = self.client.post("/api/login", json={"password": "test-password"})
        self.assertEqual(login.status_code, 200)
        events = self.client.get("/api/events")
        self.assertEqual(events.status_code, 200)
        self.assertEqual(events.json(), {"events": []})

    def test_unknown_or_malformed_figure_returns_not_found(self):
        self.assertEqual(self.client.get("/api/public/figures/nope").status_code, 404)
        self.assertEqual(
            self.client.get("/api/public/figures/not%2Fvalid").status_code,
            404,
        )


if __name__ == "__main__":
    unittest.main()
