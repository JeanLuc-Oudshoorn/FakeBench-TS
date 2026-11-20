"""Command line: generate the data, run a model, score forecasts."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .dataset import (
    SPECS,
    Spec,
    generate_dataset,
    load_dataset,
    matches_reference,
    save_dataset,
    write_reference,
)
from .distributions import FAMILIES
from .metrics import evaluate
from .models import (
    BASELINES,
    MODELS,
    load_factory,
    load_forecasts,
    run_forecaster,
    save_forecasts,
    timed_run,
)
from .report import leaderboard, write_plots

DEFAULT_DATA = "data/fakebench_v1.parquet"  # the dataset in the repository


def _option(text):
    key, separator, value = text.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(f"Expected key=value, got {text!r}")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def _generate(args):
    spec = SPECS[args.spec] if args.spec in SPECS else Spec.from_json(args.spec)
    dataset = generate_dataset(spec)
    path = save_dataset(
        dataset, args.out or f"fakebench_data/fakebench_{spec.name}.parquet"
    )
    counts = {}
    for series in dataset:
        counts[series.track] = counts.get(series.track, 0) + 1
    print(f"Wrote {len(dataset)} series to {path} ({counts})")
    if args.write_reference:
        print(f"Recorded reference values in {write_reference(dataset)}")
    matches = matches_reference(dataset)
    if matches:
        print(f"This is the reference data of spec {spec.name}.")
    elif matches is False:
        print(
            f"WARNING: this is not the reference data of spec {spec.name}. Your "
            "NumPy version draws different random numbers. The series follow the "
            "same distributions, but scores are not series-for-series comparable "
            "with published results.",
            file=sys.stderr,
        )
    return 0


def _run(args):
    dataset = load_dataset(args.data)
    options = dict(args.option)
    name = args.name or args.model.rpartition(":")[2]
    print(f"Loading {args.model} {options or ''}".rstrip())
    forecaster = load_factory(args.model)(**options)

    def progress(done, total):
        print(f"\r{name}: {done}/{total} series", end="", flush=True)

    forecasts, seconds = timed_run(
        forecaster,
        dataset,
        batch_size=args.batch_size,
        limit=args.limit,
        progress=progress,
    )
    out = args.out or Path(args.data).parent / "forecasts" / f"{name}.parquet"
    save_forecasts(
        forecasts,
        out,
        model=name,
        dataset=dataset,
        details={"factory": args.model, "options": options, "seconds": seconds},
    )
    print(f"\nWrote {len(forecasts)} forecasts to {out} in {seconds:.1f} s")
    return 0


def _score(args):
    dataset = load_dataset(args.data)
    paths = []
    default = Path(args.data).parent / "forecasts"
    for item in map(Path, args.forecasts or [default]):
        if item.is_dir():
            paths += sorted(item.glob("*.parquet"))
        elif item.is_file():
            paths.append(item)
        elif args.forecasts:
            raise SystemExit(f"No forecast file or folder at {item}")
    forecasts_by_model, submitted = {}, []
    for path in paths:
        metadata, forecasts = load_forecasts(path)
        if metadata.get("dataset_sha256") not in (None, dataset.sha256):
            raise SystemExit(f"{path} was made for a different dataset file")
        forecasts_by_model[metadata["model"]] = forecasts
        submitted.append(metadata["model"])
    covered = [set(f) for f in forecasts_by_model.values()]
    for name in args.baselines:
        if name not in forecasts_by_model:
            forecasts_by_model[name] = run_forecaster(MODELS[name](), dataset)
    if covered:
        # Baselines are scored on the same series as the submitted models.
        shared = set.intersection(*covered)
        if len(shared) < len(dataset):
            print(f"Scoring the {len(shared)} series every model forecast")
    results, per_series = evaluate(
        dataset, forecasts_by_model, bootstrap=args.bootstrap
    )
    out = Path(args.out or Path(args.data).parent / "results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    table = leaderboard(results)
    (out / "leaderboard.md").write_text(table, encoding="utf-8")
    per_series.to_parquet(out / "per_series.parquet", index=False)
    print(table)
    if not args.no_plots:
        try:
            # Plot the submitted models; with none, the reference baselines.
            highlight = submitted or list(forecasts_by_model)
            plots = write_plots(
                results, dataset, forecasts_by_model, out, highlight=highlight
            )
            print(f"Wrote {len(plots)} plots")
        except ImportError:
            print("matplotlib is not installed; skipped the plots")
    print(f"Results in {out}")
    return 0


def _list(args):
    print("Models (use with `run --model`):")
    for name, factory in MODELS.items():
        print(f"  {name:16} {factory.__doc__.splitlines()[0]}")
    print("\nStep distributions:")
    for family in FAMILIES.values():
        print(f"  {family.name:16} [{family.track}] {family.summary}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fakebench-ts",
        description="Benchmark forecasting models on random walks, where the "
        "best forecast is known exactly.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="create the benchmark data")
    generate.add_argument("--spec", default="v1", help="built-in name or JSON file")
    generate.add_argument(
        "--out", help="default: fakebench_data/fakebench_<spec>.parquet"
    )
    generate.add_argument(
        "--write-reference", action="store_true", help=argparse.SUPPRESS
    )
    generate.set_defaults(run=_generate)

    run = commands.add_parser("run", help="forecast every series with one model")
    run.add_argument("--model", required=True, help="built-in name or file.py:factory")
    run.add_argument("--data", default=DEFAULT_DATA)
    run.add_argument("--name", help="label in the results (default: model name)")
    run.add_argument("--out", help="default: <data folder>/forecasts/<name>.parquet")
    run.add_argument(
        "--option",
        action="append",
        default=[],
        type=_option,
        metavar="KEY=VALUE",
        help="passed to the model factory, e.g. device=cuda; repeatable",
    )
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--limit", type=int, help="forecast an even subset (quick trial)")
    run.set_defaults(run=_run)

    score = commands.add_parser("score", help="score forecasts and write a report")
    score.add_argument("--data", default=DEFAULT_DATA)
    score.add_argument(
        "--forecasts",
        nargs="*",
        help="forecast files or folders (default: <data folder>/forecasts)",
    )
    score.add_argument(
        "--baselines",
        nargs="*",
        default=["naive", "drift", "mean"],
        choices=BASELINES,
        help="reference forecasts to include (default: naive drift mean)",
    )
    score.add_argument("--out", help="default: <data folder>/results")
    score.add_argument("--bootstrap", type=int, default=2000)
    score.add_argument("--no-plots", action="store_true")
    score.set_defaults(run=_score)

    show = commands.add_parser("list", help="list built-in models and distributions")
    show.set_defaults(run=_list)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
