"""
RFC overview plot from an exported evaluation dataframe.

Single figure, faceted by window size (columns):
  - x-axis: algorithm (canonical order + short labels)
  - y-axis: RFC (uses the `average` column from the export by default)
  - marker shape: group_type
  - marker face color: population bias

Output: `docs/rfc_overview_scatter.pdf` by default (cwd-independent).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


CANONICAL_SLUG_ORDER: list[str] = ["H0", "H1", "A0", "A1", "A2", "S0", "S1"]
CANONICAL_SLUG_SET = set(CANONICAL_SLUG_ORDER)
SHORT_NAME_BY_SLUG: dict[str, str] = {
    "A0": "Async-Static-Grp",
    "A1": "Async-Static-Ind",
    "A2": "Async-Dyn-Ind",
    "H0": "Hybrid-Ind",
    "H1": "Hybrid-Grp-EMA",
    "S0": "Sync",
    "S1": "Sync-EMA",
}


def load_df(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path} (expected .pkl/.csv/.parquet)")


def _short_algo_label(slug: str) -> str:
    return SHORT_NAME_BY_SLUG.get(str(slug), str(slug))


def main() -> None:
    p = argparse.ArgumentParser(description="Plot RFC overview scatter from exported dataframe.")
    p.add_argument(
        "--in",
        dest="inp",
        default=None,
        help="Path to exported dataframe (.pkl/.csv/.parquet). Default: project_root/cache/cons_evaluations/exports/rfc_results_n1000_w1_3_5_10_bias0_1_2.pkl",
    )
    p.add_argument(
        "--out",
        dest="out",
        default=None,
        help="Output PDF path. Default: project_root/docs/rfc_overview_scatter.pdf",
    )
    p.add_argument(
        "--metric",
        default="average",
        help="Column to plot on Y axis (default: average, i.e. RFC in exported df).",
    )
    p.add_argument(
        "--eval-type",
        default="test",
        help="Filter eval_type (default: test).",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for horizontal jitter.")
    p.add_argument(
        "--jitter",
        type=float,
        default=0.22,
        help="Max horizontal jitter in 'category units' (default: 0.22).",
    )
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    default_in = project_root / "cache" / "cons_evaluations" / "exports" / "rfc_results_n1000_w1_3_5_10_bias0_1_2.pkl"
    inp = Path(args.inp) if args.inp else default_in

    default_out = project_root / "docs" / "rfc_overview_scatter.pdf"
    out = Path(args.out) if args.out else default_out

    df = load_df(inp)
    if "eval_type" in df.columns:
        df = df.loc[df["eval_type"] == args.eval_type].copy()
    if "algorithm_slug" in df.columns:
        df = df.loc[df["algorithm_slug"].isin(CANONICAL_SLUG_SET)].copy()

    if args.metric not in df.columns:
        raise KeyError(f"Missing metric column {args.metric!r}. Available: {sorted(df.columns)}")

    df = df.dropna(subset=[args.metric]).copy()

    slugs_in_data = set(df["algorithm_slug"].astype(str).unique().tolist())
    x_order_slugs = [s for s in CANONICAL_SLUG_ORDER if s in slugs_in_data]
    x_labels = [_short_algo_label(s) for s in x_order_slugs]
    slug_to_x = {s: i for i, s in enumerate(x_order_slugs)}

    w_sizes = sorted(df["w_size"].dropna().unique().tolist())
    if not w_sizes:
        raise ValueError("No w_size values found after filtering.")

    biases = sorted(df["bias"].dropna().unique().tolist())
    cmap = plt.get_cmap("viridis", max(3, len(biases)))
    bias_to_color = {b: cmap(i) for i, b in enumerate(biases)}

    gtypes = sorted(df["group_type"].dropna().unique().tolist())
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    gtype_to_marker = {gt: markers[i % len(markers)] for i, gt in enumerate(gtypes)}

    rng = np.random.default_rng(args.seed)

    ncols = min(4, len(w_sizes))
    nrows = int(np.ceil(len(w_sizes) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.8 * nrows),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )

    flat_axes = [ax for row in axes for ax in row]
    for ax in flat_axes[len(w_sizes) :]:
        ax.axis("off")

    for ax, w in zip(flat_axes, w_sizes):
        sub = df.loc[df["w_size"] == w]
        ax.set_title(f"W = {w}")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=35, ha="right")
        ax.set_xlabel("Algorithm")
        ax.grid(True, axis="y", alpha=0.25)

        for row in sub.itertuples(index=False):
            slug = getattr(row, "algorithm_slug")
            gt = getattr(row, "group_type")
            b = getattr(row, "bias")
            y = getattr(row, args.metric)

            x0 = float(slug_to_x[str(slug)])
            x = x0 + rng.uniform(-args.jitter, args.jitter)
            ax.scatter(
                x,
                float(y),
                s=46,
                linewidths=0.9,
                edgecolors="0.15",
                facecolors=bias_to_color[b],
                marker=gtype_to_marker[str(gt)],
                alpha=0.92,
                zorder=3,
            )

    flat_axes[0].set_ylabel("RFC")

    # Legends (matplotlib-friendly): bias colors + group_type markers
    bias_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=9, label=str(b), color=bias_to_color[b]) for b in biases
    ]
    g_handles = [
        Line2D(
            [0],
            [0],
            marker=gtype_to_marker[gt],
            linestyle="",
            markersize=9,
            markerfacecolor="0.25",
            markeredgecolor="0.15",
            label=str(gt),
        )
        for gt in gtypes
    ]

    leg_bias = fig.legend(handles=bias_handles, title="bias", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    leg_gt = fig.legend(handles=g_handles, title="group_type", loc="upper left", bbox_to_anchor=(1.02, 0.62), borderaxespad=0)
    fig.add_artist(leg_bias)

    fig.suptitle(f"RFC overview ({args.eval_type} split) — marker=group_type, color=bias", fontsize=14)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
