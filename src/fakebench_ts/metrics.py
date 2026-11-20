"""Score forecasts against the forecast that is known to be best.

All quantities are divided by the series' own typical step: the mean absolute
difference between consecutive context values (the MASE scale with lag 1).

Two kinds of measurement come out of this:

* **Error** against the values that actually followed (MASE). It is noisy,
  because the future of a random walk is noise, so it is reported as a ratio
  to the optimal forecast's error on the same series with a bootstrap interval.
* **Departure** of the forecast from the optimal forecast. Because the optimal
  forecast is known exactly, this needs no future values and has no noise from
  them: it is zero for a model that invents nothing.
"""

import numpy as np
import pandas as pd

# The recent trend a model might extrapolate is measured over this many
# horizons at the end of the context.
TREND_WINDOW_HORIZONS = 2
# Name under which the optimal forecast appears among the scored models.
OPTIMAL = "optimal"

DEFINITIONS = {
    "mase": "Mean absolute error over the horizon divided by the mean absolute "
    "one-step change in the context; averaged over series.",
    "optimal_mase": "The same for the optimal forecast (the last value on the "
    "symmetric track; the last value plus the exact median of the coming steps "
    "on the skewed track).",
    "mase_ratio": "mase / optimal_mase. 1 is the best any forecast can expect; "
    "1.10 means 10% more error than necessary. The interval is a 95% bootstrap "
    "over series.",
    "geometric_mean_ratio": "Geometric mean of each series' own error ratio.",
    "beats_optimal": "Share of series where the model had less error than the "
    "optimal forecast (ties count half). Around 0.5 or less is expected; beating "
    "it is luck.",
    "departure": "Mean absolute distance between the forecast and the optimal "
    "forecast over the horizon, in typical steps. 0 means nothing was invented.",
    "jump": "Absolute distance of the first forecast step from the last observed "
    "value.",
    "net_move": "Absolute net movement of the forecast between its first and last "
    "step: a trend or a reversion.",
    "oscillation": "Movement of the forecast beyond its net move (total up-and-down "
    "travel minus net move): cycles and wiggles. 0 for any forecast that moves in "
    "one direction only.",
    "trend_following": "How much of the context's recent trend the forecast "
    "continues: 1 means it extrapolates the trend of the last two horizons in "
    "full, 0 not at all, negative means it reverses it.",
    "mean_reversion": "How far the forecast returns to the context mean by the "
    "last step: 1 means all the way, 0 not at all.",
    "structure_r2": "Share of the variation in where forecasts end that trend "
    "following and mean reversion explain together.",
}


def mase_scale(context):
    return float(np.mean(np.abs(np.diff(context))))


def recent_slope(context, horizon):
    """Least-squares slope per step over the end of the context."""
    window = np.asarray(context[-TREND_WINDOW_HORIZONS * horizon :], dtype=float)
    steps = np.arange(len(window)) - (len(window) - 1) / 2
    return float(steps @ (window - window.mean()) / (steps @ steps))


def score_forecasts(dataset, forecasts, *, series_ids=None):
    """Per-series scores and per-step curves for one model.

    `forecasts` maps series id to the point forecast for each horizon step.
    Returns a frame with one row per series and a dict of [series, step] arrays.
    """
    rows, curves = [], {"error": [], "optimal_error": [], "departure": []}
    for series in dataset:
        if series_ids is not None and series.series_id not in series_ids:
            continue
        forecast = np.asarray(forecasts[series.series_id], dtype=float)
        if forecast.shape != (series.horizon,) or not np.isfinite(forecast).all():
            raise ValueError(
                f"Forecast for {series.series_id} must be {series.horizon} finite "
                f"values, got shape {forecast.shape}"
            )
        scale = mase_scale(series.context)
        last = series.context[-1]
        optimal = series.optimal_forecast
        distance = np.abs(forecast - optimal) / scale
        error = np.abs(series.future - forecast) / scale
        optimal_error = np.abs(series.future - optimal) / scale
        # The shape the model drew, measured from the last observed value.
        path = (forecast - last) / scale
        net_move = path[-1] - path[0]
        rows.append(
            {
                "series_id": series.series_id,
                "track": series.track,
                "family": series.family,
                "context_length": series.context_length,
                "mase": error.mean(),
                "optimal_mase": optimal_error.mean(),
                "departure": distance.mean(),
                "jump": path[0],
                "net_move": net_move,
                "oscillation": np.abs(np.diff(path)).sum() - abs(net_move),
                "end_shift": path[-1],
                # What a model would add by the last step if it extrapolated the
                # recent trend, or returned to the context mean, in full.
                "trend_pull": recent_slope(series.context, series.horizon)
                * series.horizon
                / scale,
                "mean_pull": (series.context.mean() - last) / scale,
            }
        )
        curves["error"].append(error)
        curves["optimal_error"].append(optimal_error)
        curves["departure"].append(distance)
    return pd.DataFrame(rows), {k: np.asarray(v) for k, v in curves.items()}


def _structure(frame):
    """Regress where forecasts end on the trend and mean-reversion pulls."""
    ends = frame["end_shift"].to_numpy()
    if len(ends) < 4 or np.ptp(ends) == 0:
        return {"trend_following": 0.0, "mean_reversion": 0.0, "structure_r2": 0.0}
    design = np.column_stack(
        [np.ones(len(ends)), frame["trend_pull"], frame["mean_pull"]]
    )
    coefficients, *_ = np.linalg.lstsq(design, ends, rcond=None)
    residual = ends - design @ coefficients
    return {
        "trend_following": float(coefficients[1]),
        "mean_reversion": float(coefficients[2]),
        "structure_r2": float(max(0.0, 1 - residual.var() / ends.var())),
    }


def summarise(frame, *, bootstrap=2000, seed=0):
    """Aggregate one model's per-series scores."""
    mase = frame["mase"].to_numpy()
    optimal = frame["optimal_mase"].to_numpy()
    count = len(frame)
    picks = np.random.default_rng(seed).integers(0, count, size=(bootstrap, count))
    resampled = mase[picks].mean(1) / optimal[picks].mean(1)
    low, high = np.quantile(resampled, [0.025, 0.975])
    return {
        "series": count,
        "mase": float(mase.mean()),
        "optimal_mase": float(optimal.mean()),
        "mase_ratio": float(mase.mean() / optimal.mean()),
        "mase_ratio_ci95": [float(low), float(high)],
        "geometric_mean_ratio": float(np.exp(np.mean(np.log(mase / optimal)))),
        "beats_optimal": float(np.mean((mase < optimal) + 0.5 * (mase == optimal))),
        "departure": float(frame["departure"].mean()),
        "departure_median": float(frame["departure"].median()),
        "departure_p90": float(frame["departure"].quantile(0.9)),
        "jump": float(frame["jump"].abs().mean()),
        "net_move": float(frame["net_move"].abs().mean()),
        "oscillation": float(frame["oscillation"].mean()),
        **_structure(frame),
    }


def evaluate(dataset, forecasts_by_model, *, bootstrap=2000):
    """Score several models on the series all of them forecast.

    The optimal forecast is always scored too, as the model `optimal`, so its
    own shape is on record where it is not simply the last value.

    Returns (results, per_series): a nested dict ready for JSON and one frame
    with a row per model and series.
    """
    if OPTIMAL in forecasts_by_model:
        raise ValueError(f"The model name {OPTIMAL!r} is reserved")
    forecasts_by_model = {
        OPTIMAL: {series.series_id: series.optimal_forecast for series in dataset},
        **forecasts_by_model,
    }
    shared = set.intersection(*(set(f) for f in forecasts_by_model.values()))
    if not shared:
        raise ValueError("The models have no forecast series in common")
    results = {
        "benchmark": {
            "name": dataset.spec["name"],
            "sha256": dataset.sha256,
            "horizon": dataset.horizon,
            "series": len(dataset),
            "scored_series": len(shared),
        },
        "definitions": DEFINITIONS,
        "models": {},
    }
    frames = []
    for model, forecasts in forecasts_by_model.items():
        frame, curves = score_forecasts(dataset, forecasts, series_ids=shared)
        tracks = {}
        for track, rows in frame.groupby("track", sort=True):
            index = rows.index.to_numpy()
            tracks[track] = {
                "overall": summarise(rows, bootstrap=bootstrap),
                "by_family": {
                    name: summarise(group, bootstrap=bootstrap)
                    for name, group in rows.groupby("family", sort=False)
                },
                "by_context_length": {
                    str(length): summarise(group, bootstrap=bootstrap)
                    for length, group in rows.groupby("context_length", sort=True)
                },
                "by_horizon": {
                    name: values[index].mean(0).tolist()
                    for name, values in curves.items()
                },
            }
        results["models"][model] = {"tracks": tracks}
        frames.append(frame.assign(model=model))
    return results, pd.concat(frames, ignore_index=True)
