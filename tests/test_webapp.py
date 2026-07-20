import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient

from digital_mirror.historical_catalog import catalog_root
from digital_mirror.webapp import Settings, create_app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.episode_root = Path(self.temporary.name) / "episodes"
        self.episode_root.mkdir()
        self.settings = Settings(
            episode_root=self.episode_root,
            historical_root=catalog_root(),
            schema_path=Path("schemas/replay_prediction.schema.json"),
            access_password="test-password",
            session_secret="test-session-secret",
            cookie_secure=True,
        )
        self.client = TestClient(create_app(settings=self.settings), base_url="https://testserver")

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

    def test_period_question_is_public_and_evidence_bounded(self):
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
        answer = self.client.post("/api/ask", json=payload)
        self.assertEqual(answer.status_code, 200)
        body = answer.json()
        self.assertEqual(body["figure_name"], "路德维希·范·贝多芬")
        self.assertEqual(body["period_label"], "维也纳：突破与听觉危机")
        self.assertIn("新工具", body["judgement"])
        self.assertEqual(
            body["supported_claims"][0],
            "以演奏、出版和赞助网络建立独立作曲家地位。",
        )
        self.assertEqual(body["evidence_ids"], ["beethoven-p2-e1"])
        self.assertIn("不代表人物真实说过", body["boundary_note"])

        private_prediction = self.client.post(
            "/api/predict",
            json={"episode_id": "private_event", "thinking": "enabled"},
        )
        self.assertEqual(private_prediction.status_code, 401)

    def test_personal_mirror_can_be_asked_without_exposing_raw_data(self):
        payload = {
            "question": "我在什么情况下更可能改变原来的判断？",
            "model_id": "evidence-synthesis",
            "history": [],
            "allow_cloud": False,
        }
        self.assertEqual(self.client.post("/api/me/ask", json=payload).status_code, 401)
        self.client.post("/api/login", json={"password": "test-password"})
        answer = self.client.post("/api/me/ask", json=payload)
        self.assertEqual(answer.status_code, 200)
        body = answer.json()
        self.assertEqual(body["mirror_id"], "personal")
        self.assertIn("工作假设", body["judgement"])
        self.assertNotIn("source_relative_path", str(body))

        with patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "synthetic-key"},
            clear=False,
        ):
            denied = self.client.post(
                "/api/me/ask",
                json=payload | {"model_id": "gemini"},
            )
        self.assertEqual(denied.status_code, 422)
        self.assertIn("明确授权", denied.json()["detail"])

    def test_event_replay_selects_model_and_requires_cloud_consent(self):
        source = Path(__file__).parents[1] / "templates" / "historical_episode.json"
        episode = json.loads(source.read_text(encoding="utf-8"))
        episode["episode_id"] = "synthetic_replay"
        episode["labels"]["actual_judgement"] = {
            "option_id": "clarify",
            "recorded_at": "2026-01-01T12:01:00+08:00",
            "notes": "Synthetic label.",
        }
        episode["labels"]["actual_action"] = {
            "option_id": "wait",
            "recorded_at": "2026-01-01T12:05:00+08:00",
            "notes": "Synthetic label.",
        }
        (self.episode_root / "synthetic_replay.json").write_text(
            json.dumps(episode), encoding="utf-8"
        )

        def fake_replay_runner(case, _schema, model_id, _thinking, allow_cloud):
            self.assertEqual(model_id, "gemini")
            self.assertTrue(allow_cloud)
            return (
                {
                    "episode_id": case["episode_id"],
                    "predicted_judgement": "clarify",
                    "predicted_action": "withdraw",
                    "option_probabilities": {
                        "wait": 0.45,
                        "clarify": 0.5,
                        "withdraw": 0.05,
                    },
                    "considered_option_ids": ["wait", "clarify"],
                    "tensions": ["表达 vs 风险"],
                    "unknowns": ["对方反应"],
                    "deliberation_intensity": 0.7,
                    "confidence": 0.6,
                    "evidence_ids": [],
                },
                0,
            )

        with patch.dict(os.environ, {"GEMINI_API_KEY": "synthetic-key"}, clear=False):
            client = TestClient(
                create_app(
                    settings=self.settings,
                    replay_prediction_runner=fake_replay_runner,
                ),
                base_url="https://testserver",
            )
            client.post("/api/login", json={"password": "test-password"})
            payload = {
                "episode_id": "synthetic_replay",
                "thinking": "enabled",
                "model_id": "gemini",
                "allow_cloud": False,
            }
            denied = client.post("/api/predict", json=payload)
            self.assertEqual(denied.status_code, 422)
            self.assertIn("明确授权", denied.json()["detail"])

            allowed = client.post(
                "/api/predict", json=payload | {"allow_cloud": True}
            )
            client.close()

        self.assertEqual(allowed.status_code, 200)
        body = allowed.json()
        self.assertEqual(body["model"]["model_id"], "gemini")
        self.assertEqual(body["prediction"]["predicted_action"], "withdraw")
        self.assertEqual(body["actual"]["action"], "wait")
        self.assertNotIn("source_relative_path", str(body))

    def test_unknown_or_malformed_figure_returns_not_found(self):
        self.assertEqual(self.client.get("/api/public/figures/nope").status_code, 404)
        self.assertEqual(
            self.client.get("/api/public/figures/not%2Fvalid").status_code,
            404,
        )


if __name__ == "__main__":
    unittest.main()
