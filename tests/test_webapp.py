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

        profile = self.client.get("/api/me/profile")
        self.assertEqual(profile.status_code, 200)
        self.assertEqual(profile.json()["mirror_id"], "personal")
        self.assertEqual(profile.json()["coverage"]["event_count"], 0)

    def test_period_question_is_authenticated_and_evidence_bounded(self):
        models = self.client.get("/api/public/models")
        self.assertEqual(models.status_code, 200)
        self.assertTrue(
            any(item["model_id"] == "evidence-synthesis" for item in models.json()["models"])
        )

        payload = {
            "figure_id": "ludwig-van-beethoven",
            "period_id": "beethoven-vienna-1792-1815",
            "question": "如果看到今天的生成式 AI，他可能先质疑什么？",
            "model_id": "evidence-synthesis",
            "mode": "counterfactual",
            "history": [],
        }
        self.assertEqual(self.client.post("/api/ask", json=payload).status_code, 401)
        self.client.post("/api/login", json={"password": "test-password"})
        answer = self.client.post("/api/ask", json=payload)
        self.assertEqual(answer.status_code, 200)
        body = answer.json()
        self.assertEqual(body["figure_name"], "路德维希·范·贝多芬")
        self.assertEqual(body["period_label"], "维也纳：突破与听觉危机")
        self.assertEqual(
            body["supported_claims"][0],
            "以演奏、出版和赞助网络建立独立作曲家地位。",
        )
        self.assertEqual(body["evidence_ids"], ["beethoven-p2-e1"])
        self.assertIn("不代表人物真实说过", body["boundary_note"])

    def test_unknown_or_malformed_figure_returns_not_found(self):
        self.assertEqual(self.client.get("/api/public/figures/nope").status_code, 404)
        self.assertEqual(
            self.client.get("/api/public/figures/not%2Fvalid").status_code,
            404,
        )


if __name__ == "__main__":
    unittest.main()
