"""FakeBench-TS: forecasting models on series that cannot be predicted."""

from .dataset import (
    SPECS,
    Dataset,
    Series,
    Spec,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from .distributions import FAMILIES
from .metrics import evaluate, score_forecasts, summarise
from .models import (
    MODELS,
    load_forecasts,
    run_forecaster,
    save_forecasts,
)
from .report import leaderboard, write_plots

__version__ = "0.1.0"

__all__ = [
    "FAMILIES",
    "MODELS",
    "SPECS",
    "Dataset",
    "Series",
    "Spec",
    "evaluate",
    "generate_dataset",
    "leaderboard",
    "load_dataset",
    "load_forecasts",
    "run_forecaster",
    "save_dataset",
    "save_forecasts",
    "score_forecasts",
    "summarise",
    "write_plots",
]
