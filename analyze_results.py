#!/usr/bin/env python3
"""Statistical analysis for STT benchmark results"""

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StatisticalResult:
    model: str
    dataset: str
    mean_wer: float
    std_wer: float
    ci_lower: float
    ci_upper: float
    n_samples: int


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean"""
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper


def compute_statistics(df: pd.DataFrame, n_bootstrap: int = 10000) -> list[StatisticalResult]:
    """Compute statistical summaries for each model/dataset combination"""
    results = []

    for (model, dataset), group in df.groupby(["model", "dataset"]):
        wer_values = group["wer"].values
        mean_wer = np.mean(wer_values)
        std_wer = np.std(wer_values, ddof=1)
        ci_lower, ci_upper = bootstrap_ci(wer_values, n_bootstrap)

        results.append(StatisticalResult(
            model=model,
            dataset=dataset,
            mean_wer=mean_wer,
            std_wer=std_wer,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=len(wer_values),
        ))

    return results


def compare_models(df: pd.DataFrame, model_a: str, model_b: str, dataset: str) -> dict:
    """Compare two models using paired Wilcoxon signed-rank test"""
    df_dataset = df[df["dataset"] == dataset]

    wer_a = df_dataset[df_dataset["model"] == model_a].sort_values("sample_idx")["wer"].values
    wer_b = df_dataset[df_dataset["model"] == model_b].sort_values("sample_idx")["wer"].values

    if len(wer_a) != len(wer_b):
        raise ValueError(f"Sample count mismatch: {model_a}={len(wer_a)}, {model_b}={len(wer_b)}")

    # Wilcoxon signed-rank test (non-parametric, paired)
    stat, p_value = stats.wilcoxon(wer_a, wer_b)

    # Effect size (mean difference / pooled std)
    diff = wer_a - wer_b
    mean_diff = np.mean(diff)
    effect_size = mean_diff / np.std(diff, ddof=1) if np.std(diff) > 0 else 0

    return {
        "model_a": model_a,
        "model_b": model_b,
        "dataset": dataset,
        "mean_a": np.mean(wer_a),
        "mean_b": np.mean(wer_b),
        "mean_diff": mean_diff,
        "effect_size": effect_size,
        "p_value": p_value,
    }


def holm_bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[tuple[float, bool]]:
    """Apply Holm-Bonferroni correction for multiple comparisons"""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    results = [None] * n

    for rank, idx in enumerate(sorted_indices):
        adjusted_alpha = alpha / (n - rank)
        is_significant = p_values[idx] < adjusted_alpha
        results[idx] = (p_values[idx], is_significant)

    return results


def print_summary_table(results: list[StatisticalResult]):
    """Print summary statistics table"""
    print("\n" + "=" * 90)
    print("STATISTICAL SUMMARY (95% Bootstrap CI)")
    print("=" * 90)
    print(f"{'Model':<25} {'Dataset':<10} {'WER (%)':<12} {'95% CI':<18} {'Std':<10} {'N':<6}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: (x.dataset, x.mean_wer)):
        ci_str = f"[{r.ci_lower*100:.2f}, {r.ci_upper*100:.2f}]"
        print(f"{r.model:<25} {r.dataset:<10} {r.mean_wer*100:>8.2f}%    {ci_str:<18} {r.std_wer*100:>7.2f}%   {r.n_samples:<6}")

    print("=" * 90)


def print_comparison_table(comparisons: list[dict], corrected: list[tuple[float, bool]]):
    """Print pairwise comparison table"""
    print("\n" + "=" * 100)
    print("PAIRWISE MODEL COMPARISONS (Wilcoxon signed-rank, Holm-Bonferroni corrected)")
    print("=" * 100)
    print(f"{'Model A':<20} {'Model B':<20} {'Dataset':<10} {'Diff':<12} {'Effect':<10} {'p-value':<12} {'Sig.':<6}")
    print("-" * 100)

    for comp, (p, sig) in zip(comparisons, corrected):
        diff_str = f"{comp['mean_diff']*100:+.2f}%"
        sig_str = "*" if sig else ""
        print(f"{comp['model_a']:<20} {comp['model_b']:<20} {comp['dataset']:<10} {diff_str:<12} {comp['effect_size']:>+8.3f}   {comp['p_value']:<12.4g} {sig_str:<6}")

    print("-" * 100)
    print("* = significant at α=0.05 after Holm-Bonferroni correction")
    print("Effect size: positive = Model A has higher WER (worse)")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Analyze STT benchmark results")
    parser.add_argument("input", type=Path, help="Path to detailed results CSV")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Number of bootstrap resamples")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}")
        return 1

    print(f"Loading results from: {args.input}")
    df = pd.read_csv(args.input)

    print(f"Found {len(df)} samples across {df['model'].nunique()} models and {df['dataset'].nunique()} datasets")
    print(f"Running bootstrap with {args.bootstrap} resamples...")

    # Compute summary statistics
    stat_results = compute_statistics(df, args.bootstrap)
    print_summary_table(stat_results)

    # Pairwise comparisons within each dataset
    models = df["model"].unique()
    datasets = df["dataset"].unique()

    if len(models) > 1:
        comparisons = []
        for dataset in datasets:
            for model_a, model_b in combinations(models, 2):
                try:
                    comp = compare_models(df, model_a, model_b, dataset)
                    comparisons.append(comp)
                except ValueError as e:
                    print(f"Warning: {e}")

        if comparisons:
            p_values = [c["p_value"] for c in comparisons]
            corrected = holm_bonferroni_correction(p_values, args.alpha)
            print_comparison_table(comparisons, corrected)

    return 0


if __name__ == "__main__":
    exit(main())
