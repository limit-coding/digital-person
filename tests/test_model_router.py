import json
import os
import unittest
from unittest.mock import patch

from digital_mirror.historical_catalog import load_catalog, period_from_profile
from digital_mirror.model_router import answer_historical_question


class ModelRouterTests(unittest.TestCase):
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
