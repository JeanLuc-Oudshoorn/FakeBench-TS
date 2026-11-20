"""Step distributions for the random walks.

Every family has mean zero, so the expected value of every future point of a
walk is its last observed value. The benchmark scores absolute error, whose
best forecast is the *median* of the future value, not its mean. The two
agree for the symmetric families. For the skewed families they differ, and
``median_of_sum`` gives the exact median of the sum of the next ``h`` steps.
"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import special, stats

SYMMETRIC = "symmetric"
SKEWED = "skewed"


def _zero_median(params, horizon):
    return np.zeros(horizon)


@dataclass(frozen=True)
class Family:
    """One family of zero-mean step distributions, in units of the walk's scale."""

    name: str
    track: str
    summary: str
    # Draws the family's shape parameters for one series.
    sample_params: Callable[[np.random.Generator], dict]
    # Draws ``count`` steps at scale 1.
    draw: Callable[[np.random.Generator, dict, int], np.ndarray]
    # Median of the sum of the next 1..horizon steps at scale 1.
    median_of_sum: Callable[[dict, int], np.ndarray] = _zero_median


def _no_params(std):
    return lambda rng: {"std": std}


def _student_t_params(low, high, *, log_uniform):
    def sample(rng):
        if log_uniform:
            df = math.exp(rng.uniform(math.log(low), math.log(high)))
        else:
            df = rng.uniform(low, high)
        # The variance is infinite at two degrees of freedom or fewer.
        return {"df": df, "std": math.sqrt(df / (df - 2)) if df > 2 else None}

    return sample


def _jump_params(rng):
    probability = rng.uniform(0.01, 0.1)
    size = rng.uniform(5.0, 20.0)
    return {
        "jump_probability": probability,
        "jump_size": size,
        "std": math.sqrt(1 - probability + probability * size**2),
    }


def _jump_draw(rng, params, count):
    steps = rng.standard_normal(count)
    jumps = rng.random(count) < params["jump_probability"]
    steps[jumps] *= params["jump_size"]
    return steps


def _sign(rng):
    return 1.0 if rng.random() < 0.5 else -1.0


def _gamma_params(rng):
    shape = math.exp(rng.uniform(math.log(0.5), math.log(8.0)))
    sign = _sign(rng)
    return {
        "shape": shape,
        "long_tail": "up" if sign > 0 else "down",
        "sign": sign,
        "std": math.sqrt(shape),
        "skewness": sign * 2 / math.sqrt(shape),
    }


def _gamma_draw(rng, params, count):
    return params["sign"] * (
        rng.standard_gamma(params["shape"], count) - params["shape"]
    )


def _gamma_median(params, horizon):
    # The sum of h Gamma(shape) draws is Gamma(h * shape).
    total = params["shape"] * np.arange(1, horizon + 1)
    return params["sign"] * (special.gammaincinv(total, 0.5) - total)


def _two_point_sizes(probability):
    # A rare large step one way, a frequent small step the other; variance 1.
    return (
        math.sqrt((1 - probability) / probability),
        math.sqrt(probability / (1 - probability)),
    )


def _two_point_params(rng):
    probability = rng.uniform(0.05, 0.3)
    sign = _sign(rng)
    large, small = _two_point_sizes(probability)
    return {
        "rare_step_probability": probability,
        "rare_step": sign * large,
        "common_step": -sign * small,
        "sign": sign,
        "std": 1.0,
        "skewness": sign
        * (1 - 2 * probability)
        / math.sqrt(probability * (1 - probability)),
    }


def _two_point_draw(rng, params, count):
    rare = rng.random(count) < params["rare_step_probability"]
    return np.where(rare, params["rare_step"], params["common_step"])


def _two_point_median(params, horizon):
    # The number of rare steps among h is binomial; its median fixes the sum's.
    steps = np.arange(1, horizon + 1)
    rare = stats.binom.ppf(0.5, steps, params["rare_step_probability"])
    return rare * params["rare_step"] + (steps - rare) * params["common_step"]


FAMILIES = {
    family.name: family
    for family in (
        Family(
            "normal",
            SYMMETRIC,
            "Normal steps: the textbook random walk.",
            _no_params(1.0),
            lambda rng, params, count: rng.standard_normal(count),
        ),
        Family(
            "student_t",
            SYMMETRIC,
            "Student-t steps with 3 to 30 degrees of freedom: heavier tails, finite variance.",
            _student_t_params(3.0, 30.0, log_uniform=True),
            lambda rng, params, count: rng.standard_t(params["df"], count),
        ),
        Family(
            "student_t_heavy",
            SYMMETRIC,
            "Student-t steps with 1.5 to 3 degrees of freedom: occasional very large "
            "steps; the variance is infinite at 2 or fewer.",
            _student_t_params(1.5, 3.0, log_uniform=False),
            lambda rng, params, count: rng.standard_t(params["df"], count),
        ),
        Family(
            "laplace",
            SYMMETRIC,
            "Laplace (double exponential) steps.",
            _no_params(math.sqrt(2.0)),
            lambda rng, params, count: rng.laplace(0.0, 1.0, count),
        ),
        Family(
            "logistic",
            SYMMETRIC,
            "Logistic steps.",
            _no_params(math.pi / math.sqrt(3.0)),
            lambda rng, params, count: rng.logistic(0.0, 1.0, count),
        ),
        Family(
            "uniform",
            SYMMETRIC,
            "Uniform steps on [-1, 1]: bounded, no tails.",
            _no_params(1 / math.sqrt(3.0)),
            lambda rng, params, count: rng.uniform(-1.0, 1.0, count),
        ),
        Family(
            "rademacher",
            SYMMETRIC,
            "Steps of exactly +1 or -1 with equal probability: a walk on a grid.",
            _no_params(1.0),
            lambda rng, params, count: rng.integers(0, 2, count) * 2.0 - 1.0,
        ),
        Family(
            "jump_mixture",
            SYMMETRIC,
            "Normal steps, with 1% to 10% of them 5 to 20 times larger: quiet "
            "stretches broken by sudden jumps in either direction.",
            _jump_params,
            _jump_draw,
        ),
        Family(
            "skew_gamma",
            SKEWED,
            "Centred gamma steps (shape 0.5 to 8), long tail up or down.",
            _gamma_params,
            _gamma_draw,
            _gamma_median,
        ),
        Family(
            "skew_two_point",
            SKEWED,
            "A frequent small step one way and a rare large step back (5% to 30% of "
            "steps): a sawtooth that looks like a trend with resets.",
            _two_point_params,
            _two_point_draw,
            _two_point_median,
        ),
    )
}
