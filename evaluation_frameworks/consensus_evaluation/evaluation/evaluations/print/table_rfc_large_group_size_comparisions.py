from typing import Any, Dict, List, Optional
import argparse
import math
import pandas as pd

from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.config import load_eval_res
from evaluation_frameworks.consensus_evaluation.evaluation.evaluations.print.rfc_table_metric_spec import (
    add_rfc_metric_arg,
    resolve_rfc_metric,
)
from latex_utils.latex_table_generator import LaTeXTableGeneratorSIUnitx


EVAL_TYPE: str = "test"
BIAS_KEY = 0.0

ALGOS: List[str] = [
    "eval_large_hybrid_group_updatable.py",
    "eval_large_sync_with_feedback_ema.py",
    "eval_async_with_sigmoid_policy_simple_priority_group_rec.py",
    "eval_async_with_sigmoid_policy_simple_priority_individual_rec.py",
]


def strategy_2_slug(algos: List[str]) -> Dict[str, str]:
    res = {}
    sync_ctr = 0
    async_ctr = 0
    hybrid_ctr = 0
    for algo_name in algos:
        if "async" in algo_name:
            res[f"A{async_ctr}"] = algo_name
            async_ctr += 1
        elif "hybrid" in algo_name:
            res[f"H{hybrid_ctr}"] = algo_name
            hybrid_ctr += 1
        else:
            res[f"S{sync_ctr}"] = algo_name
            sync_ctr += 1
    return res


def _safe_get_metric(d: Dict[str, Any], metric_key: str = "average") -> Optional[float]:
    if d is None:
        return math.nan
    if metric_key in d:
        try:
            return float(d[metric_key])
        except (TypeError, ValueError):
            pass
    metrics = d.get("metrics") if isinstance(d, dict) else None
    if isinstance(metrics, dict) and metric_key in metrics:
        try:
            return float(metrics[metric_key])
        except (TypeError, ValueError):
            pass
    return math.nan


def _pick_bias_block(group_data: Dict[Any, Any], bias: float) -> Dict[str, Any]:
    if not isinstance(group_data, dict):
        return {}
    if bias in group_data:
        return group_data[bias]
    bstr = str(bias)
    if bstr in group_data:
        return group_data[bstr]
    if 0 in group_data:
        return group_data[0]
    if "0" in group_data:
        return group_data["0"]
    for _, val in group_data.items():
        if isinstance(val, dict):
            return val
    return {}


def load_large_rfc_values(
    algos: List[str],
    *,
    window_size: str,
    eval_type: str,
    groups_count: int,
    group_sizes: List[int],
    metric_key: str = "average",
) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    for algo_name in algos:
        out[algo_name] = {}
        for gs in group_sizes:
            try:
                loaded = load_eval_res(
                    algo_name,
                    window_size,
                    eval_type,
                    group_size=gs,
                    groups_count=groups_count,
                )
            except Exception as e:
                print(f"WARNING: Could not load {algo_name} for group_size={gs}: {e}")
                out[algo_name][gs] = math.nan
                continue

            group_block = loaded.get("random", {}) if isinstance(loaded, dict) else {}
            bias_block = _pick_bias_block(group_block, BIAS_KEY)
            out[algo_name][gs] = _safe_get_metric(bias_block, metric_key=metric_key)
    return out


def create_table(
    values: Dict[str, Dict[int, float]],
    algos: List[str],
    *,
    group_sizes: List[int],
    adjust_A: bool = True,
    apply_async_minus_one: bool = True,
    caption: Optional[str] = None,
) -> str:
    slug2algo = strategy_2_slug(algos)
    cols = [f"size_{gs}" for gs in group_sizes]
    rows = []
    for slug, algo_name in slug2algo.items():
        is_adj = apply_async_minus_one and adjust_A and ("A" in slug)
        row: Dict[str, Any] = {"algorithm": slug + (" (RFC$_{adj.}$)" if is_adj else "")}
        for gs, c in zip(group_sizes, cols):
            val = values.get(algo_name, {}).get(gs, math.nan)
            if is_adj and pd.notna(val):
                val = val - 1
            row[c] = val
        row["average"] = pd.Series([row[c] for c in cols], dtype=float).mean(skipna=True)
        rows.append(row)

    df = pd.DataFrame(rows, columns=["algorithm"] + cols + ["average"])
    generator = LaTeXTableGeneratorSIUnitx(df, column_specs=None, column_width=1.5)
    cap = caption or rf"Porovnání large-group algoritmů skrze RFC pro různé velikosti skupin."
    return generator.generate_table(
        caption=cap,
        label="tab:large_group_size_rfc_comparison",
        cell_bold_fn=lambda row_idx, col_idx, val: (
            col_idx >= 1
            and pd.notna(val)
            and "adj." not in str(df.iloc[row_idx, 0])
            and val == df.loc[~df.iloc[:, 0].str.contains("adj.", regex=False), df.columns[col_idx]].min(skipna=True)
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", default=10, help="Window size")
    parser.add_argument("--groups-count", type=int, default=100, help="Evaluation groups count used in cache path")
    parser.add_argument("--group-sizes", nargs="+", type=int, default=[3, 5, 7, 10], help="Group sizes to compare")
    add_rfc_metric_arg(parser)
    args = parser.parse_args()

    spec = resolve_rfc_metric(args.rfc_metric)
    print(
        f"[table_rfc_large_group_size] W={args.window_size} metric={args.rfc_metric} "
        f"→ `{spec.storage_key}`"
    )

    vals = load_large_rfc_values(
        ALGOS,
        window_size=str(args.window_size),
        eval_type=EVAL_TYPE,
        groups_count=args.groups_count,
        group_sizes=list(args.group_sizes),
        metric_key=spec.storage_key,
    )
    print(
        create_table(
            vals,
            ALGOS,
            group_sizes=list(args.group_sizes),
            apply_async_minus_one=spec.subtract_one_for_async_slug,
            caption=rf"Porovnání large-group algoritmů — {spec.latex_caption_cs} (různé velikosti skupin).",
        )
    )
