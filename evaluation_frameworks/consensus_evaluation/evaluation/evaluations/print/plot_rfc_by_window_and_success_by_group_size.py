from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.export_rfc_dataframe import (
    build_dataframe,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _canon_algo_name(df: pd.DataFrame) -> pd.Series:
    if "algorithm_short_name" in df.columns:
        return df["algorithm_short_name"].astype(str)
    if "algorithm_slug" in df.columns:
        return df["algorithm_slug"].astype(str)
    return df["algorithm_file"].astype(str)


def plot_rfc_vs_window(
    df: pd.DataFrame,
    *,
    windows: List[int] | None,
    biases: List[float] | None,
    groups_count: int | None,
    output: Path,
) -> None:
    d = df.copy()
    d["algorithm"] = _canon_algo_name(d)
    if windows:
        d = d[d["w_size"].isin(windows)]
    if biases:
        d = d[d["bias"].isin(biases)]
    if groups_count is not None:
        d = d[d["groups_count"] == groups_count]
    # "basic" setup: no explicit large-group slice
    d = d[d["group_size"].isna()]
    d = d[d["average"].notna()]
    if d.empty:
        raise ValueError("No rows for RFC-vs-window plot after filters.")

    g = (
        d.groupby(["algorithm", "w_size"], as_index=False)["average"]
        .mean()
        .rename(columns={"average": "rfc"})
        .sort_values(["algorithm", "w_size"])
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.2), layout="constrained")
    for algo, part in g.groupby("algorithm"):
        ax.plot(part["w_size"], part["rfc"], "-o", linewidth=2.0, markersize=6, label=algo)
    ax.set_xlabel("Velikost okna ω")
    ax.set_ylabel("RFC (průměr kol do první shody)")
    ax.set_title("RFC vs velikost okna")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_success_vs_group_size(
    df: pd.DataFrame,
    *,
    windows: List[int] | None,
    biases: List[float] | None,
    groups_count: int | None,
    output: Path,
) -> None:
    d = df.copy()
    d["algorithm"] = _canon_algo_name(d)
    if windows:
        d = d[d["w_size"].isin(windows)]
    if biases:
        d = d[d["bias"].isin(biases)]
    if groups_count is not None:
        d = d[d["groups_count"] == groups_count]
    d = d[d["group_size"].notna()]
    d = d[d["success_rate"].notna()]
    if d.empty:
        raise ValueError("No rows for success-vs-group-size plot after filters.")

    g = (
        d.groupby(["algorithm", "group_size"], as_index=False)["success_rate"]
        .mean()
        .sort_values(["algorithm", "group_size"])
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.2), layout="constrained")
    for algo, part in g.groupby("algorithm"):
        ax.plot(part["group_size"], part["success_rate"], "-o", linewidth=2.0, markersize=6, label=algo)
    ax.set_xlabel("Velikost skupiny")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Success rate vs velikost skupiny")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate RFC-vs-window and success-vs-group-size plots from pickles.")
    p.add_argument(
        "--cache-root",
        default=str(_repo_root() / "cache" / "cons_evaluations"),
        help="Cache root with cons_evaluations pickles.",
    )
    p.add_argument("--all-runs", action="store_true", help="Use all numbered runs (default: latest run per eval dir).")
    p.add_argument("--windows", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--biases", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    p.add_argument("--groups-count", type=int, default=1000)
    p.add_argument("--out-dir", default=str(_repo_root() / "img"))
    p.add_argument("--prefix", default="rfc_success")
    args = p.parse_args()

    df = build_dataframe(cache_root=Path(args.cache_root), latest_only=not args.all_runs)
    if df.empty:
        raise ValueError(f"No rows found in cache root: {args.cache_root}")

    out_dir = Path(args.out_dir)
    p1 = out_dir / f"{args.prefix}_rfc_vs_window.png"
    p2 = out_dir / f"{args.prefix}_success_vs_group_size.png"

    plot_rfc_vs_window(
        df,
        windows=list(args.windows),
        biases=list(args.biases),
        groups_count=args.groups_count,
        output=p1,
    )
    plot_success_vs_group_size(
        df,
        windows=list(args.windows),
        biases=list(args.biases),
        groups_count=args.groups_count,
        output=p2,
    )
    print(f"Saved {p1}")
    print(f"Saved {p2}")


if __name__ == "__main__":
    main()

