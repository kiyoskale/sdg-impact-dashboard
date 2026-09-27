import math
import unittest

import pandas as pd

from src.analytics import (
    add_growth_columns,
    build_correlation_matrix,
    build_wide_frame,
    calculate_normalized_growth,
    compute_summary_stats,
    shared_years,
)


def series(values, years, code="X", country="A"):
    return pd.DataFrame(
        {"country_name": country, "indicator_code": code, "year": years, "value": values}
    )


class SummaryStatsTest(unittest.TestCase):
    def test_descriptive_statistics(self):
        stats = compute_summary_stats(series([10.0, 20.0, 30.0, 40.0], [2000, 2001, 2002, 2003]))
        self.assertEqual(stats["observations"], 4)
        self.assertAlmostEqual(stats["mean"], 25.0)
        self.assertAlmostEqual(stats["std"], math.sqrt(500 / 3))
        self.assertAlmostEqual(stats["median"], 25.0)
        self.assertAlmostEqual(stats["total_growth_pct"], 300.0)
        self.assertEqual(stats["peak_growth_year"], 2001)
        self.assertAlmostEqual(stats["peak_growth_pct"], 100.0)

    def test_growth_is_annualised_across_gaps(self):
        frame = add_growth_columns(series([100.0, 121.0], [2000, 2002]))
        self.assertAlmostEqual(frame["growth_pct"].iloc[1], 10.0)

    def test_most_volatile_year_uses_rolling_window(self):
        stats = compute_summary_stats(
            series([1.0, 1.0, 1.0, 10.0, 1.0], [2000, 2001, 2002, 2003, 2004]), window=3
        )
        self.assertEqual(stats["most_volatile_year"], 2003)

    def test_single_observation_has_no_growth_or_spread(self):
        stats = compute_summary_stats(series([5.0], [2010]))
        self.assertEqual(stats["observations"], 1)
        self.assertIsNone(stats["peak_growth_year"])
        self.assertIsNone(stats["most_volatile_year"])
        self.assertTrue(math.isnan(stats["std"]))

    def test_zero_base_does_not_produce_infinite_growth(self):
        frame = add_growth_columns(series([0.0, 5.0], [2000, 2001]))
        self.assertTrue(math.isnan(frame["growth_pct"].iloc[1]))


class CorrelationTest(unittest.TestCase):
    years = [2000, 2001, 2002, 2003]

    def test_perfect_positive_and_negative_pearson(self):
        corr = build_correlation_matrix(
            [
                series([1.0, 2.0, 3.0, 4.0], self.years, code="A"),
                series([2.0, 4.0, 6.0, 8.0], self.years, code="B"),
                series([4.0, 3.0, 2.0, 1.0], self.years, code="C"),
            ]
        )
        self.assertAlmostEqual(corr.loc["A", "B"], 1.0)
        self.assertAlmostEqual(corr.loc["A", "C"], -1.0)

    def test_spearman_captures_monotonic_non_linear_relationship(self):
        frames = [
            series([1.0, 2.0, 3.0, 4.0], self.years, code="A"),
            series([1.0, 8.0, 27.0, 125.0], self.years, code="B"),
        ]
        self.assertLess(build_correlation_matrix(frames).loc["A", "B"], 1.0)
        self.assertAlmostEqual(
            build_correlation_matrix(frames, method="spearman").loc["A", "B"], 1.0
        )

    def test_merge_aligns_on_year_and_respects_min_periods(self):
        a = series([1.0, 2.0, 3.0, 4.0], [2000, 2001, 2002, 2003], code="A")
        b = series([3.0, 5.0, 7.0, 9.0], [2001, 2002, 2003, 2004], code="B")
        wide = build_wide_frame([a, b])
        self.assertEqual(list(wide.index), [2000, 2001, 2002, 2003, 2004])
        self.assertAlmostEqual(build_correlation_matrix([a, b]).loc["A", "B"], 1.0)
        self.assertTrue(math.isnan(build_correlation_matrix([a, b], min_periods=4).loc["A", "B"]))

    def test_empty_series_are_skipped(self):
        empty = series([], [], code="E")
        wide = build_wide_frame([series([1.0, 2.0], [2000, 2001], code="A"), empty])
        self.assertEqual(list(wide.columns), ["A"])


class BenchmarkTest(unittest.TestCase):
    frame = pd.concat(
        [
            series([50.0, 75.0], [2000, 2001], country="A"),
            series([200.0, 100.0], [2000, 2001], country="B"),
            series([10.0], [2001], country="C"),
        ]
    )

    def test_growth_relative_to_base_year(self):
        growth = calculate_normalized_growth(self.frame, 2000).set_index(["country_name", "year"])
        self.assertAlmostEqual(growth.loc[("A", 2000), "growth_pct"], 0.0)
        self.assertAlmostEqual(growth.loc[("A", 2001), "growth_pct"], 50.0)
        self.assertAlmostEqual(growth.loc[("B", 2001), "growth_pct"], -50.0)

    def test_countries_missing_base_year_are_dropped(self):
        growth = calculate_normalized_growth(self.frame, 2000)
        self.assertNotIn("C", set(growth["country_name"]))

    def test_shared_years(self):
        self.assertEqual(shared_years(self.frame), [2001])


if __name__ == "__main__":
    unittest.main()
