from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.load_exported_rfc_dataframe import (
    CANONICAL_SLUG_ORDER,
    SHORT_NAME_BY_SLUG,
    load_df,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_out_dir() -> Path:
    return _repo_root() / "img"


def _prepare_matrix(
    df: pd.DataFrame,
    *,
    windows: List[int] | None,
    biases: List[float] | None,
    group_types: List[str] | None,
    groups_counts: List[int] | None,
    include_large_groups: bool,
) -> pd.DataFrame:
    d = df.copy()
    d = d[d["algorithm_slug"].isin(CANONICAL_SLUG_ORDER)]
    d = d[d["average"].notna()]

    if windows:
        d = d[d["w_size"].isin(windows)]
    if biases:
        d = d[d["bias"].isin(biases)]
    if group_types:
        d = d[d["group_type"].isin(group_types)]
    if groups_counts:
        d = d[d["groups_count"].isin(groups_counts)]

    if not include_large_groups:
        d = d[d["group_size"].isna()]

    # Condition key = one data point in cloud/clustering.
    d["cond_key"] = (
        "w="
        + d["w_size"].astype(str)
        + "|b="
        + d["bias"].astype(str)
        + "|gt="
        + d["group_type"].astype(str)
        + "|gsize="
        + d["group_size"].fillna("base").astype(str)
    )

    p = (
        d.groupby(["cond_key", "algorithm_slug"], as_index=False)["average"]
        .mean()
        .pivot(index="cond_key", columns="algorithm_slug", values="average")
    )
    p = p[[c for c in CANONICAL_SLUG_ORDER if c in p.columns]]
    p = p.dropna(axis=0, how="any")
    if p.empty:
        raise ValueError("No complete rows for selected filters (matrix is empty).")
    p = p.rename(columns=lambda s: SHORT_NAME_BY_SLUG.get(str(s), str(s)))
    return p


def _plot_heatmap(matrix: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(matrix.columns)), max(6, 0.22 * len(matrix.index))), layout="constrained")
    sns.heatmap(matrix, cmap="viridis_r", ax=ax, cbar_kws={"label": "RFC (nižší je lepší)"})
    ax.set_title("RFC heatmap: podmínky × algoritmy")
    ax.set_xlabel("Algoritmus")
    ax.set_ylabel("Podmínka (w|bias|group_type|group_size)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_corr(matrix: pd.DataFrame, output: Path) -> None:
    corr = matrix.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Spearman ρ"})
    ax.set_title("Korelační mapa algoritmů (Spearman)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_cluster_cloud(matrix: pd.DataFrame, output: Path, n_clusters: int) -> None:
    # rows = conditions, cols = algorithm RFC features
    X = matrix.values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(Xs)

    k = max(2, min(n_clusters, len(matrix)))
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(Xs)

    plot_df = pd.DataFrame(
        {"PC1": XY[:, 0], "PC2": XY[:, 1], "cluster": labels.astype(str), "cond_key": matrix.index}
    )
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="cluster", palette="tab10", s=75, ax=ax)
    ax.set_title(
        "Cluster-like cloud podmínek (PCA + KMeans)\n"
        f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.2f}, PC2={pca.explained_variance_ratio_[1]:.2f}"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Annotate a few extreme points for interpretability.
    for idx in plot_df.nlargest(4, "PC1").index.tolist() + plot_df.nsmallest(4, "PC1").index.tolist():
        r = plot_df.loc[idx]
        ax.annotate(r["cond_key"], (r["PC1"], r["PC2"]), fontsize=7, alpha=0.8)

    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Pattern mining visuals (heatmap, correlation, PCA+cluster cloud) from exported RFC dataframe.")
    p.add_argument("--in", dest="inp", required=True, help="Input dataframe (.pkl/.csv/.parquet) from export_rfc_dataframe.py")
    p.add_argument("--out-dir", default=None, help="Output directory for figures (default: <repo>/img).")
    p.add_argument("--prefix", default="rfc_patterns", help="Filename prefix.")
    p.add_argument("--windows", type=int, nargs="+", default=None)
    p.add_argument("--biases", type=float, nargs="+", default=None)
    p.add_argument("--group-types", nargs="+", default=None)
    p.add_argument("--groups-counts", type=int, nargs="+", default=None)
    p.add_argument("--include-large-groups", action="store_true")
    p.add_argument("--clusters", type=int, default=4, help="KMeans cluster count for cloud.")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(Path(args.inp))
    matrix = _prepare_matrix(
        df,
        windows=args.windows,
        biases=args.biases,
        group_types=args.group_types,
        groups_counts=args.groups_counts,
        include_large_groups=args.include_large_groups,
    )

    heat_path = out_dir / f"{args.prefix}_heatmap.pdf"
    corr_path = out_dir / f"{args.prefix}_corr.pdf"
    cloud_path = out_dir / f"{args.prefix}_cluster_cloud.pdf"

    _plot_heatmap(matrix, heat_path)
    _plot_corr(matrix, corr_path)
    _plot_cluster_cloud(matrix, cloud_path, n_clusters=args.clusters)

    print(f"Saved {heat_path}")
    print(f"Saved {corr_path}")
    print(f"Saved {cloud_path}")


if __name__ == "__main__":
    main()

