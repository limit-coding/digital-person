import json
import os
import unittest
from unittest.mock import patch

from digital_mirror.historical_catalog import load_catalog, period_from_profile
from digital_mirror.model_router import (
    CloudConsentRequiredError,
    answer_historical_question,
    answer_replay_prediction,
)


class ModelRouterTests(unittest.TestCase):
    def test_replay_prediction_requires_consent_and_preserves_option_contract(self):
        case = {
            "schema_version": 1,
            "episode_id": "test_replay",
            "task_mode": "next_action",
            "cutoff_at": "2026-01-01T12:00:00+08:00",
            "domain": "other",
            "question": "发送还是等待？",
            "options": [
                {"option_id": "send", "description": "发送"},
                {"option_id": "hold", "description": "等待"},
            ],
            "persona_state": {},
            "evidence": [
                {
                    "evidence_id": "e1",
                    "observed_at": "2026-01-01T11:00:00+08:00",
                    "author_role": "self",
                    "summary": "已经完成草稿。",
                }
            ],
            "instructions": [],
            "required_output_fields": [
                "episode_id",
                "predicted_judgement",
                "predicted_action",
                "option_probabilities",
                "considered_option_ids",
                "tensions",
                "unknowns",
                "deliberation_intensity",
                "confidence",
                "evidence_ids",
            ],
        }
        prediction = {
            "episode_id": "test_replay",
            "predicted_judgement": "send",
            "predicted_action": "hold",
            "option_probabilities": {"send": 0.45, "hold": 0.55},
            "considered_option_ids": ["send", "hold"],
            "tensions": ["表达 vs 风险"],
            "unknowns": ["对方反应"],
            "deliberation_intensity": 0.7,
            "confidence": 0.6,
            "evidence_ids": ["e1"],
        }

        def fake_request(_url, _payload, _api_key, _timeout):
            return {
                "candidates": [
                    {"content": {"parts": [{"text": json.dumps(prediction)}]}}
                ]
            }

        with patch.dict(os.environ, {"GEMINI_API_KEY": "synthetic-key"}, clear=False):
            with self.assertRaises(CloudConsentRequiredError):
                answer_replay_prediction(case, {}, "gemini", "enabled", False)
            result, retries = answer_replay_prediction(
                case,
                {"type": "object"},
                "gemini",
                "enabled",
                True,
                request=fake_request,
            )

        self.assertEqual(result["predicted_judgement"], "send")
        self.assertEqual(result["predicted_action"], "hold")
        self.assertEqual(retries, 0)

    def test_model_claims_cannot_promote_themselves_to_evidence(self):
        profile = load_catalog()["ludwig-van-beethoven"]
        period = period_from_profile(profile, "beethoven-vienna-1792-1815")
        self.assertIsNotNone(period)

        def fake_request(_url, _payload, _api_key, _timeout):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "judgement": "更像一种新工具",
                                    "answer": "测试回答",
                                    "supported_claims": ["模型自行补充的事实"],
                                    "speculative_claims": ["反事实推演"],
                                    "unknowns": ["未知"],
                                    "evidence_ids": ["invented-evidence"],
                                    "follow_up_questions": [],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        with patch.dict(
            os.environ,
            {
                "DIGITAL_MIRROR_COMPATIBLE_API_KEY": "synthetic-key",
                "DIGITAL_MIRROR_COMPATIBLE_BASE_URL": "https://example.invalid/v1",
                "DIGITAL_MIRROR_COMPATIBLE_MODEL": "synthetic-model",
            },
            clear=False,
        ):
            result = answer_historical_question(
                profile,
                period,
                "一个开放问题",
                "compatible",
                "counterfactual",
                [],
                request=fake_request,
            )

        self.assertEqual(result["supported_claims"], period["anchors"])
        self.assertEqual(result["judgement"], "更像一种新工具")
        self.assertIn("模型自行补充的事实", result["speculative_claims"])
        self.assertNotIn("invented-evidence", result["evidence_ids"])
        self.assertEqual(result["evidence_ids"], ["beethoven-p2-e1"])


if __name__ == "__main__":
    unittest.main()
