import unittest

from refresh_paper_performance import render
from sample_random_claim_fpr import sample_matches
from evaluate_fresh_mlp_fpr import binomial_summary


class PerformanceTableTest(unittest.TestCase):
    def rows(self, quality=1.05):
        return [("Generator", "Data", "O5", [
            {"wm_acc": 80., "clean_fp": 30., "fid_ratio": quality} for _ in range(5)])]

    def test_performance_reports_randomized_fpr_not_five_target_matches(self):
        rows = self.rows()
        for row in rows[0][3]:
            row["randomized_fpr"] = binomial_summary(310, 10000)
        output = render(rows)
        self.assertIn("$80.0{\\pm}0.0$ & 3.100", output)
        self.assertNotIn(" & 30.000 & ", output)
        self.assertIn(" & FPR (", output)
        self.assertNotIn("Randomized FPR", output)
        self.assertNotIn("score threshold", output)
        self.assertNotIn("FNR", output)
        self.assertNotIn("Expected FPR", output)
        self.assertNotIn("analytical baseline", output)
        self.assertNotIn(" & Queries", output)
        self.assertNotIn(" & Metric", output)

    def test_caption_does_not_repeat_shared_evaluation_protocol(self):
        output = render(self.rows())
        caption = output.split(r"\caption{", 1)[1].split(r"\label", 1)[0]
        self.assertIn("Recovery, clean-model FPR, and quality preservation", caption)
        self.assertNotIn("population standard deviation", caption)
        self.assertNotIn("Protocol details", caption)
        self.assertNotIn("uniform message claims", caption)

    def test_small_nonzero_fpr_is_not_rounded_to_zero(self):
        rows = self.rows()
        for row in rows[0][3]:
            row["randomized_fpr"] = binomial_summary(1, 10000)
        self.assertIn(" & 0.010 & ", render(rows))

    def test_unmeasured_randomized_fpr_is_not_replaced_with_fixed_target_rate(self):
        self.assertIn(r"$80.0{\pm}0.0$ & \textemdash", render(self.rows()))

    def test_h32_table_does_not_present_analytical_fpr_as_performance(self):
        data = self.rows()
        data[0] = ("Generator", "Data", "H32", data[0][3])
        self.assertNotIn("2.328", render(data))

    def test_claim_sampling_uses_all_codewords_even_for_constant_clean_output(self):
        result = sample_matches(["111"] * 20, 3, 10000, 42)
        self.assertEqual(result["expected_fpr_pct"], 12.5)
        self.assertLess(abs(result["sampled_fpr_pct"] - 12.5), 1.)
        self.assertEqual(result["new_velocity_queries"], 0)

    def test_sampling_rejects_wrong_payload_and_reports_probability_units(self):
        with self.assertRaises(ValueError):
            sample_matches(["00"], 3, 100, 42)
        result = sample_matches(["0" * 32], 32, 100, 42)
        self.assertEqual(result["expected_fpr_probability"], 1 / 2**32)
        self.assertEqual(result["expected_fpr_pct"], 100 / 2**32)

    def test_zero_matches_report_uncertainty_not_zero_population_probability(self):
        result = sample_matches(["0" * 32], 32, 100000, 42)
        self.assertEqual(result["matches"], 0)
        self.assertAlmostEqual(result["zero_match_one_sided_95_upper_pct"], 0.0029956874)
        self.assertGreater(result["wilson_95_interval_pct"][1], 0)

    def test_main_table_preserves_zero_observed_and_checks_match_counts(self):
        rows = self.rows()
        for row in rows[0][3]:
            row["randomized_fpr"] = binomial_summary(0, 10000)
        output = render(rows)
        self.assertIn("0 (observed)", output)
        self.assertNotIn("Fixed-target range", output)
        self.assertNotIn("score threshold", output)
        self.assertNotIn("Expected FPR", output)
        for row in rows[0][3]:
            row["randomized_fpr"]["fpr_pct"] = 1
        with self.assertRaises(ValueError):
            render(rows)

    def test_missing_quality_never_becomes_zero_or_a_measured_ratio(self):
        output = render(self.rows(None))
        self.assertIn("\\textemdash", output)
        self.assertIn("pending", output)
        with self.assertRaises(ValueError):
            render(self.rows(None), require_complete=True)

    def test_missing_messages_and_invalid_metrics_are_rejected(self):
        data = self.rows()
        data[0][3].pop()
        with self.assertRaises(ValueError):
            render(data)
        for invalid in (float("nan"), -1.):
            with self.assertRaises(ValueError):
                render(self.rows(invalid))


if __name__ == "__main__":
    unittest.main()
