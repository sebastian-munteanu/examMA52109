## Demonstrates hierarchical clustering and compares k choices
## with agglomerative clustering on a difficult dataset,
## saving evaluation metrics and plots to files.

from __future__ import annotations

import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster_maker.agglomerative import agglomerative_clustering
from cluster_maker.evaluation import compute_inertia, silhouette_score_sklearn
from cluster_maker.plotting_clustered import plot_clusters_2d
from cluster_maker.preprocessing import select_features, standardise_features

OUTPUT_DIR = "demo_output"

def evaluate_k_grid(X: np.ndarray, k_values: List[int]) -> pd.DataFrame:
    #Compute silhouette and centroid-based inertia for each candidate k.
    records = []
    for k in k_values:
        labels, centroids = agglomerative_clustering(X, k=k)
        sil = silhouette_score_sklearn(X, labels)
        inertia = compute_inertia(X, labels, centroids)
        records.append({"k": k, "silhouette": sil, "inertia": inertia})
    return pd.DataFrame(records)


def plot_k_comparison(results_df: pd.DataFrame, best_k: int, base_name: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Silhouette vs k
    axes[0].plot(results_df["k"], results_df["silhouette"], marker="o", linewidth=2.5)
    axes[0].axvline(best_k, color="red", linestyle="--", linewidth=2, label=f"Best k={best_k}")
    axes[0].set_xlabel("k (clusters)")
    axes[0].set_ylabel("Silhouette score")
    axes[0].set_title("Silhouette vs k")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Inertia-style metric vs k (using centroids for spread indication)
    axes[1].plot(results_df["k"], results_df["inertia"], marker="s", linewidth=2.5, color="darkorange")
    axes[1].axvline(best_k, color="red", linestyle="--", linewidth=2, label=f"Best k={best_k}")
    axes[1].set_xlabel("k (clusters)")
    axes[1].set_ylabel("Inertia (centroid-based)")
    axes[1].set_title("Within-cluster spread vs k")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_agglomerative_k_comparison.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(args: List[str]) -> None:
    # Require exactly one CSV path, similar to cluster_plot.py behavior
    if len(args) != 1:
        print("Usage: python cluster_plot.py <input_csv>")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Agglomerative Clustering Demo ===")
    print(f"Data: {input_path}\n")

    df = pd.read_csv(input_path)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        print("Error: need at least 2 numeric columns.")
        sys.exit(1)

    # Use up to 4 numeric features to give hierarchical clustering more structure
    selected_features = numeric_cols[:4]
    print(f"Using features: {selected_features}\n")

    feature_df = select_features(df, selected_features)
    X = standardise_features(feature_df.to_numpy(dtype=float))

    k_values = list(range(2, 8))  # modest grid of cluster counts
    print(f"Evaluating k in {k_values} ...")
    results_df = evaluate_k_grid(X, k_values)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Save metrics for each k
    metrics_csv = os.path.join(OUTPUT_DIR, f"{base_name}_agglomerative_metrics.csv")
    results_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics CSV: {metrics_csv}")

    # Choose best k by silhouette (higher is better)
    best_idx = results_df["silhouette"].idxmax()
    best_k = int(results_df.loc[best_idx, "k"])
    best_sil = results_df.loc[best_idx, "silhouette"]
    print("\nResults:")
    print(results_df.to_string(index=False))
    print(f"\nBest k: {best_k} (silhouette={best_sil:.4f})\n")

    # Save comparison plot
    comparison_plot = plot_k_comparison(results_df, best_k, base_name)
    print(f"Saved comparison plot: {comparison_plot}")

    # Fit final model and plot clusters (first two dimensions only for display)
    labels, centroids = agglomerative_clustering(X, k=best_k)
    fig, _ = plot_clusters_2d(
        X[:, :2],
        labels,
        centroids=centroids[:, :2],
        title=f"Agglomerative clustering (k={best_k})",
    )
    cluster_plot = os.path.join(OUTPUT_DIR, f"{base_name}_agglomerative_clusters.png")
    fig.savefig(cluster_plot, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved cluster plot: {cluster_plot}")
    print("\nDemo complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
