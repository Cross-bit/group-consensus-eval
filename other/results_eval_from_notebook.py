# Auto-generated from other/ResultsEval.ipynb

# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_pickle("cache/cons_evaluations/exports/rfc_results_n1000_w1_3_5_10_bias0_1_2.pkl")
df = df.loc[df.eval_type == "test"]
df = df.loc[~df.algorithm_slug.str.contains("\?")]
df

# %%
df.average.unique()

# %%
df.columns

# %%
df.shape

# %%
df = df.dropna(subset="average")
df.shape

# %%
df.loc[df.w_size == 5].groupby(["w_size","bias","algorithm_slug"])[["first_consensus_global_position_across_groups"]].mean()

# %%
df.loc[df.w_size == 5].groupby(["w_size","bias","group_type","algorithm_slug"])[["first_consensus_global_position_across_groups"]].mean()

# %%
df.groupby(["w_size","bias","group_type"])[["average"]].mean()

# %%
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path

# %%
dfx = df.groupby(["w_size","bias","group_type"])[["average"]].mean().reset_index()
dfx["avg_w"] = dfx["average"]* dfx["w_size"]
g = sns.FacetGrid(dfx, col="bias", legend_out=True, height=2.5, aspect=1.2 )
g.map_dataframe(sns.lineplot, x="w_size", y="avg_w", hue="group_type", style="group_type", markers=True,  linewidth=2)
g.add_legend(
    title="Group Type",
    bbox_to_anchor=(1.00, 0.20),
    loc="lower right")



g.axes[0,0].set_ylabel(r'$RFC * \omega$')
g.axes[0,0].set_xlabel(r'$\omega$')
g.axes[0,1].set_xlabel(r'$\omega$')
g.axes[0,2].set_xlabel(r'$\omega$')

g.axes[0,0].set_xticks([1,3,5,10])
g.axes[0,1].set_xticks([1,3,5,10])
g.axes[0,2].set_xticks([1,3,5,10])

g.set_titles("")

# add titles inside each subplot
for ax, title in zip(g.axes.flat, g.col_names):
    ax.text(
        0.15, 0.99, f"Bias: {title}",   # position (x,y) in axes coords
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11
    )

plt.tight_layout()
project_root = Path(__file__).resolve().parents[1]
out_dir = project_root / "docs"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "results_per_grouptype.pdf", bbox_inches="tight")
