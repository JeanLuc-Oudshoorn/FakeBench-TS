"""Leaderboard tables and plots from scored results."""

import numpy as np

from .metrics import OPTIMAL

TRACK_NOTES = {
    "symmetric": "Step distributions are symmetric around zero, so the optimal "
    "forecast is the last observed value at every step (the same as `naive`).",
    "skewed": "Step distributions have mean zero but are skewed. The last value "
    "is still the expected future value, but the forecast with the lowest "
    "absolute error is the median of the future value, which lies a little away "
    "from the last value. Models are measured against that median.",
}

# Categorical colours in a fixed order; a model keeps its colour across plots.
SERIES_COLOURS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
SURFACE, INK, INK_SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def leaderboard(results):
    """Markdown: one table per track, best error ratio first."""
    info = results["benchmark"]
    lines = [
        f"# FakeBench-TS {info['name']} results",
        "",
        f"{info['scored_series']} of {info['series']} series scored, "
        f"{info['horizon']} forecast steps each. All distances are in typical "
        "steps of the series (its mean absolute one-step change).",
    ]
    tracks = sorted({t for m in results["models"].values() for t in m["tracks"]})
    for track in reversed(tracks):  # symmetric first
        rows = sorted(
            (
                (name, model["tracks"][track]["overall"])
                for name, model in results["models"].items()
                if track in model["tracks"]
            ),
            key=lambda item: (
                item[0] != OPTIMAL,
                item[1]["mase_ratio"],
                item[1]["departure"],
            ),
        )
        lines += [
            "",
            f"## {track.capitalize()} track ({rows[0][1]['series']} series)",
            "",
            TRACK_NOTES.get(track, ""),
            "",
            "| Model | Error vs optimal (95% interval) | MASE | Beats optimal | "
            "Departure | Jump | Net move | Oscillation | Trend following | "
            "Mean reversion |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, row in rows:
            low, high = row["mase_ratio_ci95"]
            label = "*optimal forecast*" if name == OPTIMAL else name
            # Adding 0.0 turns a rounded -0.0 into 0.0.
            shape = " | ".join(
                f"{round(row[key], 2) + 0.0:.2f}"
                for key in (
                    "departure",
                    "jump",
                    "net_move",
                    "oscillation",
                    "trend_following",
                    "mean_reversion",
                )
            )
            lines.append(
                f"| {label} | {row['mase_ratio']:.3f} ({low:.3f} to {high:.3f}) "
                f"| {row['mase']:.2f} | {row['beats_optimal']:.0%} | {shape} |"
            )
        lengths = sorted(
            {
                int(k)
                for _, m in results["models"].items()
                for k in m["tracks"].get(track, {}).get("by_context_length", {})
            }
        )
        lines += [
            "",
            "By length of the history the model was given: error vs optimal, "
            "with departure in brackets.",
            "",
            "| Model | " + " | ".join(f"{n} points" for n in lengths) + " |",
            "|---|" + "---|" * len(lengths),
        ]
        for name, _ in rows:
            if name == OPTIMAL:
                continue
            groups = results["models"][name]["tracks"][track]["by_context_length"]
            cells = (
                f"{groups[str(n)]['mase_ratio']:.3f} ({groups[str(n)]['departure']:.2f})"
                for n in lengths
            )
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## What the columns mean", ""]
    labels = {
        "mase_ratio": "Error vs optimal",
        "mase": "MASE",
        "beats_optimal": "Beats optimal",
        "departure": "Departure",
        "jump": "Jump",
        "net_move": "Net move",
        "oscillation": "Oscillation",
        "trend_following": "Trend following",
        "mean_reversion": "Mean reversion",
    }
    lines += [
        f"- **{label}**: {results['definitions'][key]}" for key, label in labels.items()
    ]
    return "\n".join(lines) + "\n"


def _axes(plt, width, height, **options):
    figure, axes = plt.subplots(figsize=(width, height), facecolor=SURFACE, **options)
    for axis in np.atleast_1d(axes).ravel():
        axis.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(AXIS)
        axis.tick_params(colors=MUTED, length=0, labelsize=9)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
    return figure, axes


def _save(figure, path):
    figure.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    return path


def write_plots(results, dataset, forecasts_by_model, directory, *, highlight=None):
    """Write PNG plots for each track. Returns the paths written.

    `highlight` names the models to draw (default: all, at most eight). The
    optimal forecast is the reference in every plot, not one of the models.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Helvetica Neue", "DejaVu Sans"]
    names = [
        n
        for n in (highlight or results["models"])
        if n in results["models"] and n != OPTIMAL
    ][: len(SERIES_COLOURS)]
    if not names:
        return []
    colours = dict(zip(names, SERIES_COLOURS))
    written = []
    tracks = sorted({t for n in names for t in results["models"][n]["tracks"]})
    for track in tracks:
        scored = {n: results["models"][n]["tracks"][track] for n in names}

        # Extra error over the optimal forecast: one hue, sorted, with intervals.
        order = sorted(names, key=lambda n: scored[n]["overall"]["mase_ratio"])
        figure, axis = _axes(plt, 7.5, 0.5 * len(order) + 1.4)
        values = [100 * (scored[n]["overall"]["mase_ratio"] - 1) for n in order]
        bounds = [
            [100 * (b - 1) for b in scored[n]["overall"]["mase_ratio_ci95"]]
            for n in order
        ]
        positions = np.arange(len(order))[::-1]
        axis.barh(positions, values, height=0.5, color=SERIES_COLOURS[0])
        for position, value, (low, high) in zip(positions, values, bounds):
            axis.plot([low, high], [position] * 2, color=INK_SECONDARY, linewidth=1.2)
            axis.annotate(
                f"{value:+.1f}%",
                (max(high, value), position),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=INK,
            )
        axis.axvline(0, color=AXIS, linewidth=1)
        axis.set_yticks(positions, order)
        axis.tick_params(axis="y", labelcolor=INK, labelsize=10)
        axis.grid(False)
        axis.grid(axis="x", color=GRID, linewidth=0.8)
        axis.margins(x=0.15)
        axis.set_xlabel(
            "Extra error over the optimal forecast (%), with 95% interval",
            color=INK_SECONDARY,
            fontsize=9,
        )
        axis.set_title(
            f"Cost of forecasting structure that is not there ({track} track)",
            loc="left",
            color=INK,
            fontsize=11,
        )
        written.append(_save(figure, directory / f"error_vs_optimal_{track}.png"))
        plt.close(figure)

        # Distance from the optimal forecast at each step ahead.
        figure, axis = _axes(plt, 7.5, 4.2)
        for name in names:
            curve = scored[name]["by_horizon"]["departure"]
            steps = np.arange(1, len(curve) + 1)
            axis.plot(steps, curve, color=colours[name], linewidth=2, label=name)
            if len(names) <= 4:
                axis.annotate(
                    name,
                    (steps[-1], curve[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    color=INK,
                )
        axis.set_xlim(1, len(curve))
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Forecast step", color=INK_SECONDARY, fontsize=9)
        axis.set_ylabel(
            "Mean distance from the optimal forecast\n(typical steps of the series)",
            color=INK_SECONDARY,
            fontsize=9,
        )
        axis.set_title(
            f"How far forecasts stray from the optimal forecast ({track} track)",
            loc="left",
            color=INK,
            fontsize=11,
        )
        if len(names) > 1:
            axis.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
        written.append(_save(figure, directory / f"departure_by_horizon_{track}.png"))
        plt.close(figure)

        # Example forecasts: one series per family, the middle context length.
        examples, seen = [], set()
        lengths = sorted({s.context_length for s in dataset})
        middle = lengths[len(lengths) // 2]
        for series in dataset:
            if (
                series.track == track
                and series.context_length == middle
                and series.family not in seen
                and all(series.series_id in forecasts_by_model[n] for n in names)
            ):
                seen.add(series.family)
                examples.append(series)
        if not examples:
            continue
        columns = 2
        rows = -(-len(examples) // columns)
        figure, grid = _axes(
            plt, 11, 2.6 * rows + 0.6, nrows=rows, ncols=columns, squeeze=False
        )
        for axis, series in zip(grid.ravel(), examples):
            shown = min(series.context_length, 3 * series.horizon)
            past = np.arange(-shown + 1, 1)
            ahead = np.arange(1, series.horizon + 1)
            axis.plot(past, series.context[-shown:], color=INK, linewidth=1.2)
            axis.plot(
                ahead,
                series.future,
                color=MUTED,
                linewidth=1.2,
                label="what happened",
            )
            axis.plot(
                ahead,
                series.optimal_forecast,
                color=INK,
                linewidth=1.2,
                linestyle=(0, (4, 3)),
                label="optimal forecast",
            )
            for name in names:
                axis.plot(
                    ahead,
                    forecasts_by_model[name][series.series_id],
                    color=colours[name],
                    linewidth=2,
                    label=name,
                )
            axis.axvline(0.5, color=AXIS, linewidth=1)
            axis.set_title(series.family, loc="left", color=INK, fontsize=10)
        for axis in grid.ravel()[len(examples) :]:
            axis.set_visible(False)
        handles, labels = grid[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 6),
            frameon=False,
            fontsize=9,
            labelcolor=INK,
            bbox_to_anchor=(0.5, 1.0),
        )
        figure.suptitle(
            f"Example forecasts, context of {middle} points ({track} track); "
            "forecasts start at step 1",
            x=0.01,
            y=1.04,
            ha="left",
            color=INK,
            fontsize=11,
        )
        figure.tight_layout()
        written.append(_save(figure, directory / f"examples_{track}.png"))
        plt.close(figure)
    return written
