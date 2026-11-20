# FakeBench-TS

FakeBench-TS tests what a time-series foundation model does with a series that
cannot be predicted. Every series is a random walk: each value is the previous
value plus a random step drawn from a distribution with mean zero. The past
says nothing about the direction of the next step, so the best forecast is
known exactly. Any trend, cycle or return to the mean in a model's forecast is
something the model made up, and the benchmark measures how much of it there
is and what it costs.

## What the best forecast is

The benchmark scores absolute error (MAE, reported as MASE). The forecast that
minimises expected absolute error is the **median** of the future value. The
forecast that minimises squared error is its **mean**.

- For a zero-mean random walk the mean of every future value is the last
  observed value, whatever the shape of the step distribution.
- The median is the last observed value only if the step distribution is
  symmetric. With skewed steps the median of the coming steps is not zero, so
  under absolute error the last value is slightly worse than the optimum.

The benchmark therefore has two tracks, reported separately:

| Track | Step distributions | Optimal forecast |
|---|---|---|
| **symmetric** (headline) | symmetric around zero | the last observed value, at every step |
| **skewed** | mean zero, skewed | the last value plus the exact median of the sum of the coming steps |

On the skewed track the median is known in closed form for both families, so
models are still measured against the true optimum.

Absolute error is also the right choice for the heavy-tailed families: a
Student-t distribution with two or fewer degrees of freedom has a mean but no
finite variance, so squared error would not settle down.

## The series

`fakebench-ts list` prints this table from the code.

| Family | Track | Steps |
|---|---|---|
| `normal` | symmetric | normal |
| `student_t` | symmetric | Student-t, 3 to 30 degrees of freedom |
| `student_t_heavy` | symmetric | Student-t, 1.5 to 3 degrees of freedom (variance infinite at 2 or fewer) |
| `laplace` | symmetric | Laplace |
| `logistic` | symmetric | logistic |
| `uniform` | symmetric | uniform on [-1, 1] |
| `rademacher` | symmetric | exactly +1 or -1 |
| `jump_mixture` | symmetric | normal, with 1% to 10% of steps 5 to 20 times larger |
| `skew_gamma` | skewed | centred gamma, shape 0.5 to 8, long tail up or down |
| `skew_two_point` | skewed | a frequent small step one way, a rare large step back |

Each series draws its own shape parameters, a step size between 0.01 and 1000
(log-uniform) and, for half the series, a starting level away from zero.

The built-in dataset `v1` has 2,100 series: 1,500 symmetric and 600 skewed,
split evenly over context lengths of 128, 512 and 2,048 points, each followed
by 64 points to forecast. The two Student-t families have double weight.

## What is measured

Everything is divided by the series' typical step: the mean absolute change
between consecutive context values (the MASE scale).

**Error against what happened.** `MASE` is the usual scaled absolute error.
The future of a random walk is noise, so MASE alone says little. The headline
is **error vs optimal**: the model's mean MASE divided by the optimal
forecast's mean MASE on the same series, with a 95% bootstrap interval. It is
1 for the optimal forecast, and no forecast can expect less than 1.

**Departure from the optimal forecast.** Because the optimal forecast is known,
the benchmark can measure the distance between it and the model's forecast
directly, without using the future values at all. This has none of the noise
of the error measure. `Departure` is that distance averaged over the horizon.
It is zero for a model that invents nothing.

**The shape of the forecast**, measured from the last observed value:

- `Jump`: how far the first forecast step is from the last value.
- `Net move`: how far the forecast travels between its first and last step.
- `Oscillation`: up-and-down travel beyond the net move. Zero for a forecast
  that only moves in one direction; large for an invented cycle.
- `Trend following`: how much of the recent trend in the context the forecast
  continues (1 = extrapolates it in full).
- `Mean reversion`: how far the forecast returns to the context mean
  (1 = all the way).

The last two come from one regression over all series of where the forecast
ends on where a full trend extrapolation and a full return to the mean would
end. They can be near zero for a model that still departs a lot: that means
its departures are not explained by either mechanism.

Reference forecasts calibrate the numbers: `naive` (last value), `drift`
(average change over the context, continued), `mean` (context mean), `linear`
(recent trend, continued) and `seasonal_naive` (last cycle, repeated).

## Install

```bash
pip install -e .            # numpy, scipy, pandas, pyarrow
pip install -e ".[plots]"   # plus matplotlib for the plots
```

Or run it from a checkout without installing: `PYTHONPATH=src python -m fakebench_ts ...`.

## Use

```bash
fakebench-ts generate                         # writes fakebench_data/fakebench_v1.parquet
fakebench-ts run --model chronos2 --option device=cuda
fakebench-ts run --model timestral --option device=cuda
fakebench-ts score                            # leaderboard, results.json, plots
```

`run` writes one file of forecasts per model to `fakebench_data/forecasts/`.
`score` reads every file there, adds the reference forecasts, and writes
`leaderboard.md`, `results.json` (also broken down by family, context length
and forecast step), `per_series.parquet` and plots to `fakebench_data/results/`.
`run --limit 100` forecasts an even subset for a quick trial.

Built-in adapters: `chronos2`, `chronos_bolt`, `timestral`. They need the
model's own package (`chronos-forecasting`, `timestral`). Pass options such as
`model_id`, `revision`, `device` and `cache_dir` with `--option key=value`.

### Your own model

Write a factory that returns a function from contexts to point forecasts:

```python
# my_model.py
import numpy as np


def build(device="cpu"):
    model = ...  # load your model

    def forecast(contexts, horizon):
        # contexts: list of 1-D float arrays of equal length
        # return: array [len(contexts), horizon], the median forecast
        return np.stack([model.predict(c, horizon) for c in contexts])

    return forecast
```

```bash
fakebench-ts run --model my_model.py:build --name my-model --option device=cuda
```

Return the model's median (0.5 quantile), since absolute error is what is
scored. From Python the same steps are `load_dataset`, `run_forecaster`,
`evaluate` and `leaderboard` in `fakebench_ts`.

## Reproducibility

A dataset is fully determined by its spec (seed, families, lengths). Each
series has its own random stream, so adding a family does not change the
others. A custom spec is a JSON file with any of the fields of `Spec` in
`dataset.py`, passed as `generate --spec my_spec.json`.

The same NumPy version gives the same series on Windows and Linux up to
rounding in the last digits (about 1e-13 of a step, from differently rounded
logarithms). NumPy does not promise the same random numbers across versions,
so the package stores the end values `v1` produced when it was defined (NumPy
2.5) and `generate` reports whether a new file still matches them. If it does
not, the series follow the same distributions but scores are not
series-for-series comparable with published ones.

Forecast files record the checksum of the dataset file they were made from,
and `score` refuses forecasts made from a different file.

## Limits

- Each series is forecast once, from the end of its context. Uncertainty in
  the error ratio comes from the number of series, which the interval reflects.
- Only point forecasts are scored. Whether a model's quantiles widen with the
  horizon the way a random walk's do is not measured yet.
- Steps are independent and identically distributed. Walks with changing
  volatility are also unpredictable in direction, and are not included.

## Tests

```bash
python -m unittest discover -s tests
```
