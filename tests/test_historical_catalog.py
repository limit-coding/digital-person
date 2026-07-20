import unittest

from digital_mirror.historical_catalog import catalog_root, load_catalog, public_summary


class HistoricalCatalogTests(unittest.TestCase):
    def test_global_catalog_loads_isolated_profiles(self):
        catalog = load_catalog()
        self.assertGreaterEqual(len(catalog), 11)
        self.assertIn("mao-zedong", catalog)
        self.assertIn("mahatma-gandhi", catalog)
        self.assertIn("nelson-mandela", catalog)
        self.assertIn("simon-bolivar", catalog)
        self.assertIn("liliuokalani", catalog)
        self.assertIn("mustafa-kemal-ataturk", catalog)
        self.assertIn("rachel-carson", catalog)
        for figure_id, profile in catalog.items():
            self.assertEqual(catalog_root().joinpath(figure_id).name, figure_id)
            self.assertEqual(profile["figure_id"], figure_id)
            self.assertTrue(profile["slices"])
            for slice_ in profile["slices"]:
                self.assertAlmostEqual(
                    sum(item["probability"] for item in slice_["hypotheses"]),
                    1.0,
                )
                evidence_ids = {
                    item["evidence_id"] for item in slice_["evidence"]
                }
                for hypothesis in slice_["hypotheses"]:
                    self.assertTrue(set(hypothesis["evidence_ids"]) <= evidence_ids)

    def test_public_summary_does_not_embed_evidence(self):
        profile = load_catalog()["charles-darwin"]
        summary = public_summary(profile)
        self.assertNotIn("slices", summary)
        self.assertNotIn("evidence", summary)
        self.assertEqual(summary["slice_count"], 1)


if __name__ == "__main__":
    unittest.main()
