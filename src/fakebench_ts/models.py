"""Forecasters: reference baselines, adapters for published models, and the runner.

A forecaster is any callable ``forecast(contexts, horizon)`` that takes a list
of equal-length 1-D float arrays and returns one point forecast per context,
as an array of shape ``[len(contexts), horizon]``. The point forecast should
be the model's median, because the benchmark scores absolute error.

A factory builds a forecaster from keyword options. ``MODELS`` maps names to
factories; ``load_factory`` also accepts ``path/to/file.py:factory`` or
``package.module:factory`` for models that are not built in.
"""

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .dataset import METADATA_KEY
from .metrics import recent_slope

# ---------------------------------------------------------------- baselines


def naive():
    """Repeat the last value. Optimal on the symmetric track."""
    return lambda contexts, horizon: np.repeat(
        [[c[-1]] for c in contexts], horizon, axis=1
    )


def mean():
    """Forecast the context mean: full reversion to the mean."""
    return lambda contexts, horizon: np.repeat(
        [[np.mean(c)] for c in contexts], horizon, axis=1
    )


def _line(last, slope, horizon):
    return last + slope * np.arange(1, horizon + 1)


def drift():
    """Continue the average change over the whole context (random walk with drift)."""
    return lambda contexts, horizon: np.array(
        [_line(c[-1], (c[-1] - c[0]) / (len(c) - 1), horizon) for c in contexts]
    )


def linear():
    """Continue the least-squares trend of the last two horizons from the last value."""
    return lambda contexts, horizon: np.array(
        [_line(c[-1], recent_slope(c, horizon), horizon) for c in contexts]
    )


def seasonal_naive(period=24):
    """Repeat the last `period` values: an invented cycle."""
    period = int(period)
    return lambda contexts, horizon: np.array(
        [np.resize(c[-period:], horizon) for c in contexts]
    )


# ----------------------------------------------------------------- adapters


def _device(device):
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def chronos2(
    model_id="amazon/chronos-2",
    revision=None,
    device=None,
    batch_size=32,
    cache_dir=None,
):
    """Chronos-2 (needs `chronos-forecasting` 2.x). Point forecast: median."""
    import torch
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(
        model_id,
        revision=revision,
        device_map=_device(device),
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
    )

    def forecast(contexts, horizon):
        quantiles, _ = pipeline.predict_quantiles(
            [{"target": np.asarray(c, dtype=np.float32)} for c in contexts],
            prediction_length=horizon,
            quantile_levels=[0.5],
            batch_size=int(batch_size),
            cross_learning=False,
        )
        return np.stack([q[0, :, 0].float().cpu().numpy() for q in quantiles])

    return forecast


def chronos_bolt(
    model_id="amazon/chronos-bolt-base", revision=None, device=None, cache_dir=None
):
    """Chronos-Bolt (needs `chronos-forecasting`). Point forecast: median."""
    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(
        model_id, revision=revision, device_map=_device(device), cache_dir=cache_dir
    )

    def forecast(contexts, horizon):
        quantiles, _ = pipeline.predict_quantiles(
            torch.tensor(np.asarray(contexts), dtype=torch.float32),
            prediction_length=horizon,
            quantile_levels=[0.5],
        )
        return quantiles[:, :, 0].float().cpu().numpy()

    return forecast


def timestral(
    model_id="mistralai/Timestral-209M",
    revision=None,
    device=None,
    max_context=None,
    cache_dir=None,
):
    """Timestral (needs the `timestral` package). Point forecast: median."""
    from timestral import TimestralForecaster

    forecaster = TimestralForecaster.from_pretrained(
        model_id,
        revision=revision,
        device=_device(device),
        max_context=None if max_context is None else int(max_context),
        cache_dir=cache_dir,
    )

    def forecast(contexts, horizon):
        result = forecaster.forecast(
            np.asarray(contexts, dtype=np.float32), prediction_length=horizon
        )
        median = np.asarray(result.median.float().cpu())
        return median.reshape(len(contexts), horizon)

    return forecast


MODELS = {
    "naive": naive,
    "mean": mean,
    "drift": drift,
    "linear": linear,
    "seasonal_naive": seasonal_naive,
    "chronos2": chronos2,
    "chronos_bolt": chronos_bolt,
    "timestral": timestral,
}
BASELINES = ("naive", "mean", "drift", "linear", "seasonal_naive")


def load_factory(name):
    """A built-in name, `file.py:factory` or `package.module:factory`."""
    if name in MODELS:
        return MODELS[name]
    location, separator, attribute = name.rpartition(":")
    if not separator:
        raise ValueError(
            f"Unknown model {name!r}. Built in: {', '.join(MODELS)}. For your own "
            "model pass path/to/file.py:factory or package.module:factory."
        )
    if location.endswith(".py"):
        path = Path(location).resolve()
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(location)
    return getattr(module, attribute)


# ------------------------------------------------------------------- runner


def run_forecaster(forecaster, dataset, *, batch_size=64, limit=None, progress=None):
    """Forecast every series from its context alone. Returns {series id: forecast}.

    Series are sent in batches of one context length. `limit` keeps an evenly
    spaced subset (for a quick trial); models then only compare on that subset.
    """
    selected = list(dataset)
    if limit is not None and limit < len(selected):
        picks = np.linspace(0, len(selected) - 1, int(limit)).round().astype(int)
        selected = [selected[i] for i in sorted(set(picks))]
    by_length = {}
    for series in selected:
        by_length.setdefault(series.context_length, []).append(series)
    forecasts, done = {}, 0
    for group in by_length.values():
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            # A copy, so a model cannot change the data it is scored against.
            contexts = [series.context.copy() for series in batch]
            output = np.asarray(forecaster(contexts, dataset.horizon), dtype=float)
            if output.shape != (len(batch), dataset.horizon):
                raise ValueError(
                    f"The forecaster returned shape {output.shape}; expected "
                    f"{(len(batch), dataset.horizon)}"
                )
            if not np.isfinite(output).all():
                raise ValueError("The forecaster returned values that are not finite")
            forecasts.update(zip((s.series_id for s in batch), output))
            done += len(batch)
            if progress:
                progress(done, len(selected))
    return forecasts


def save_forecasts(forecasts, path, *, model, dataset, details=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "series_id": list(forecasts),
            "forecast": pa.array(list(forecasts.values()), pa.list_(pa.float64())),
        }
    )
    metadata = {"model": model, "dataset_sha256": dataset.sha256, **(details or {})}
    pq.write_table(
        table.replace_schema_metadata({METADATA_KEY: json.dumps(metadata)}), path
    )
    return path


def load_forecasts(path):
    """Returns (metadata, {series id: forecast})."""
    table = pq.read_table(path)
    metadata = json.loads((table.schema.metadata or {}).get(METADATA_KEY, b"{}"))
    metadata.setdefault("model", Path(path).stem)
    forecasts = dict(
        zip(
            table.column("series_id").to_pylist(),
            (np.asarray(f) for f in table.column("forecast").to_pylist()),
        )
    )
    return metadata, forecasts


def timed_run(forecaster, dataset, **options):
    started = perf_counter()
    forecasts = run_forecaster(forecaster, dataset, **options)
    return forecasts, perf_counter() - started
