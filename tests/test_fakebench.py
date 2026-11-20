"""Run with `python -m unittest discover -s tests` (or pytest)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fakebench_ts import (  # noqa: E402
    FAMILIES,
    MODELS,
    Spec,
    evaluate,
    generate_dataset,
    load_dataset,
    run_forecaster,
    save_dataset,
)
from fakebench_ts.cli import main  # noqa: E402
from fakebench_ts.dataset import (  # noqa: E402
    SPECS,
    generate_series,
    matches_reference,
)
from fakebench_ts.distributions import SKEWED, SYMMETRIC  # noqa: E402

SMALL = Spec(name="small", seed=3, context_lengths=(160, 400), series_per_weight=20)


class Distributions(unittest.TestCase):
    def draws(self, family, count=400_000):
        rng = np.random.default_rng(11)
        params = family.sample_params(rng)
        return params, family.draw(rng, params, count)

    def test_symmetric_families_are_symmetric_around_zero(self):
        for family in FAMILIES.values():
            if family.track != SYMMETRIC:
                continue
            with self.subTest(family.name):
                params, steps = self.draws(family)
                self.assertAlmostEqual(
                    np.mean(steps > 0) + np.mean(steps == 0) / 2, 0.5, 2
                )
                low, high = np.quantile(steps, [0.1, 0.9])
                self.assertAlmostEqual(low / high, -1.0, 1)
                self.assertTrue((family.median_of_sum(params, 8) == 0).all())

    def test_finite_variance_families_have_mean_zero_and_stated_spread(self):
        for family in FAMILIES.values():
            params, steps = self.draws(family)
            if params["std"] is None or family.name == "student_t_heavy":
                continue
            with self.subTest(family.name):
                self.assertLess(abs(steps.mean()), 0.01 * params["std"])
                self.assertAlmostEqual(steps.std() / params["std"], 1.0, 1)

    def test_skewed_families_state_the_median_of_the_coming_steps(self):
        for family in FAMILIES.values():
            if family.track != SKEWED:
                continue
            with self.subTest(family.name):
                rng = np.random.default_rng(5)
                params = family.sample_params(rng)
                steps = family.draw(rng, params, 200_000 * 20).reshape(-1, 20)
                sums = np.cumsum(steps, axis=1)
                stated = family.median_of_sum(params, 20)
                self.assertTrue(np.any(stated != 0))
                for h in (1, 5, 20):
                    # The stated value must split the simulated sums in half.
                    # The margin absorbs rounding where sums take few values.
                    below = np.mean(sums[:, h - 1] < stated[h - 1] - 1e-9)
                    at_most = np.mean(sums[:, h - 1] <= stated[h - 1] + 1e-9)
                    self.assertLessEqual(below, 0.51)
                    self.assertGreaterEqual(at_most, 0.49)


class Data(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(SMALL)

    def test_same_spec_gives_same_data(self):
        self.assertEqual(self.dataset.sha256, generate_dataset(SMALL).sha256)

    def test_a_series_does_not_depend_on_the_rest_of_the_spec(self):
        alone = generate_series(SMALL, "laplace", 400, 7)
        match = next(s for s in self.dataset if s.series_id == alone.series_id)
        np.testing.assert_array_equal(alone.context, match.context)
        np.testing.assert_array_equal(alone.future, match.future)

    def test_counts_shapes_and_tracks(self):
        weights = sum(SMALL.families.values())
        self.assertEqual(len(self.dataset), weights * 20 * 2)
        for series in self.dataset:
            self.assertIn(series.context_length, (160, 400))
            self.assertEqual(series.future.shape, (112,))
            self.assertEqual(series.optimal_offset.shape, (112,))
            self.assertTrue(np.isfinite(series.context).all())
            self.assertEqual(series.context[0], series.start)
            if series.track == SYMMETRIC:
                self.assertTrue((series.optimal_offset == 0).all())

    def test_file_round_trip_is_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            path = save_dataset(self.dataset, Path(folder) / "data.parquet")
            loaded = load_dataset(path)
        self.assertEqual(loaded.sha256, self.dataset.sha256)
        self.assertEqual(loaded.spec, self.dataset.spec)
        self.assertEqual(loaded.series[5].params, self.dataset.series[5].params)

    def test_built_in_specs_produce_their_reference_data(self):
        # A NumPy version that draws different numbers is a warning for users,
        # but this test pins the versions the benchmark is developed on.
        for name, spec in SPECS.items():
            self.assertIs(matches_reference(generate_dataset(spec)), True, name)

    def test_the_dataset_in_the_repository_is_the_reference_data(self):
        path = Path(__file__).resolve().parents[1] / "data" / "fakebench_v1.parquet"
        published = load_dataset(path)
        self.assertEqual(published.spec, SPECS["v1"].to_dict())
        self.assertIs(matches_reference(published), True)

    def test_changed_data_does_not_match_the_reference(self):
        self.assertIsNone(matches_reference(self.dataset))
        changed = generate_dataset(SPECS["v1"])
        series = changed.series[40]
        series.future[-1] += 1e-6 * series.scale
        self.assertIs(matches_reference(changed), False)


class Scores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(SMALL)
        forecasts = {
            name: run_forecaster(MODELS[name](), cls.dataset)
            for name in ("naive", "mean", "drift", "linear", "seasonal_naive")
        }
        cls.forecasts = forecasts
        cls.results, cls.per_series = evaluate(cls.dataset, forecasts, bootstrap=200)

    def overall(self, model, track):
        return self.results["models"][model]["tracks"][track]["overall"]

    def test_naive_is_exactly_optimal_on_the_symmetric_track(self):
        row = self.overall("naive", SYMMETRIC)
        self.assertEqual(row["mase_ratio"], 1.0)
        self.assertEqual(row["mase_ratio_ci95"], [1.0, 1.0])
        self.assertEqual(row["beats_optimal"], 0.5)
        for key in ("departure", "jump", "net_move", "oscillation", "structure_r2"):
            self.assertEqual(row[key], 0.0)

    def test_naive_is_not_optimal_on_the_skewed_track(self):
        self.assertGreater(self.overall("naive", SKEWED)["departure"], 0.05)
        row = self.overall("optimal", SKEWED)
        self.assertEqual(row["departure"], 0.0)
        self.assertEqual(row["mase_ratio"], 1.0)
        # The optimal forecast's own shape is on record: it leaves the last value.
        self.assertGreater(row["jump"], 0.05)
        self.assertEqual(self.overall("optimal", SYMMETRIC)["jump"], 0.0)

    def test_the_optimal_forecast_has_the_lowest_expected_error(self):
        # More series than a single run scores, so luck averages out.
        dataset = generate_dataset(
            Spec(
                name="skew",
                seed=9,
                context_lengths=(128,),
                series_per_weight=1500,
                families={"skew_gamma": 1, "skew_two_point": 1},
            )
        )
        naive = run_forecaster(MODELS["naive"](), dataset)
        results, _ = evaluate(dataset, {"naive": naive}, bootstrap=500)
        row = results["models"]["naive"]["tracks"][SKEWED]["overall"]
        self.assertGreater(row["mase_ratio_ci95"][0], 1.0)

    def test_the_name_of_the_optimal_forecast_is_reserved(self):
        with self.assertRaises(ValueError):
            evaluate(self.dataset, {"optimal": self.forecasts["naive"]})

    def test_invented_structure_costs_error(self):
        for model in ("mean", "drift", "linear", "seasonal_naive"):
            with self.subTest(model):
                row = self.overall(model, SYMMETRIC)
                self.assertGreater(row["mase_ratio_ci95"][0], 1.0)
                self.assertLess(row["beats_optimal"], 0.5)
                self.assertGreater(row["departure"], 0.5)

    def test_structure_measures_identify_the_reference_forecasts(self):
        trend = self.overall("linear", SYMMETRIC)
        self.assertAlmostEqual(trend["trend_following"], 1.0)
        self.assertAlmostEqual(trend["mean_reversion"], 0.0)
        self.assertAlmostEqual(trend["structure_r2"], 1.0)
        self.assertAlmostEqual(trend["oscillation"], 0.0)
        revert = self.overall("mean", SYMMETRIC)
        self.assertAlmostEqual(revert["mean_reversion"], 1.0)
        self.assertAlmostEqual(revert["trend_following"], 0.0)
        self.assertAlmostEqual(revert["net_move"], 0.0)
        self.assertGreater(revert["jump"], 1.0)
        cycle = self.overall("seasonal_naive", SYMMETRIC)
        self.assertGreater(cycle["oscillation"], 10.0)

    def test_horizon_curves_and_groups(self):
        track = self.results["models"]["drift"]["tracks"][SYMMETRIC]
        self.assertEqual(len(track["by_horizon"]["departure"]), 112)
        self.assertEqual(self.results["benchmark"]["horizons"], [32, 64, 112])
        matrix = track["by_horizon_length"]
        self.assertEqual(sorted(matrix), ["112", "32", "64"])
        self.assertEqual(sorted(matrix["32"]), ["160", "400", "all"])
        self.assertEqual(matrix["112"]["all"], track["overall"])
        self.assertEqual(matrix["112"]["400"], track["by_context_length"]["400"])
        # A cut scores the first steps of the same forecast.
        self.assertAlmostEqual(
            matrix["32"]["all"]["departure"],
            float(np.mean(track["by_horizon"]["departure"][:32])),
        )
        self.assertLess(
            matrix["32"]["all"]["departure"], matrix["64"]["all"]["departure"]
        )
        self.assertEqual(sorted(track["by_context_length"]), ["160", "400"])
        self.assertIn("student_t_heavy", track["by_family"])
        # Drift departs further from the last value at every step.
        self.assertTrue(np.all(np.diff(track["by_horizon"]["departure"]) > 0))
        self.assertEqual(len(self.per_series), 6 * len(self.dataset))

    def test_bad_forecasts_are_rejected(self):
        with self.assertRaises(ValueError):
            run_forecaster(lambda c, h: np.zeros((len(c), h - 1)), self.dataset)
        with self.assertRaises(ValueError):
            run_forecaster(lambda c, h: np.full((len(c), h), np.nan), self.dataset)


class CommandLine(unittest.TestCase):
    def test_generate_run_score(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            spec = folder / "spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "name": "tiny",
                        "context_lengths": [96, 200],
                        "series_per_weight": 4,
                        "families": {"normal": 1, "student_t": 1, "skew_gamma": 1},
                    }
                )
            )
            data = folder / "tiny.parquet"
            model = folder / "my_model.py"
            model.write_text(
                "import numpy as np\n"
                "def build(damping=0.5):\n"
                "    return lambda contexts, horizon: np.array(\n"
                "        [np.full(horizon, c[-1] * damping) for c in contexts])\n"
            )
            self.assertEqual(
                main(["generate", "--spec", str(spec), "--out", str(data)]), 0
            )
            self.assertEqual(
                main(
                    [
                        "run",
                        "--data",
                        str(data),
                        "--model",
                        "seasonal_naive",
                        "--option",
                        "period=12",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "run",
                        "--data",
                        str(data),
                        "--model",
                        f"{model}:build",
                        "--name",
                        "damped",
                        "--option",
                        "damping=0.9",
                        "--limit",
                        "10",
                    ]
                ),
                0,
            )
            self.assertEqual(main(["score", "--data", str(data)]), 0)
            results = json.loads((folder / "results" / "results.json").read_text())
            self.assertEqual(results["benchmark"]["scored_series"], 10)
            self.assertEqual(
                set(results["models"]),
                {"optimal", "damped", "seasonal_naive", "naive", "drift", "mean"},
            )
            self.assertIn(
                "| damped |", (folder / "results" / "leaderboard.md").read_text()
            )
            self.assertTrue((folder / "results" / "per_series.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
