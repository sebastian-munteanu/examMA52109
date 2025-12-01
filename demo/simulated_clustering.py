## This script loads a CSV, picks the first 2 numeric columns,
## runs k-means clustering for k=2 to 6, evaluates silhouette scores,
## finds the best clustering and summarizes the results including visualizations.

from __future__ import annotations

import sys
import os
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker.interface import run_clustering
from cluster_maker.evaluation import silhouette_score_sklearn

OUTPUT_DIR = "demo_output"


def main(args: List[str]) -> None:
    if len(args) != 1:
        print("Usage: python demo/simulated_clustering.py <input_csv>")
        sys.exit(1)

    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Optimal Clustering Analysis ===\n")
    print(f"Input file: {input_path}")
    
    # Load and inspect data
    df = pd.read_csv(input_path)
    print(f"Loaded: {df.shape[0]} samples, {df.shape[1]} columns\n")
    
    # Extract numeric features
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_cols) < 2:
        print("Error: Need at least 2 numeric columns.")
        sys.exit(1)
    
    selected_features = numeric_cols[:2]
    print(f"Selected features: {selected_features}\n")

    # Evaluate different cluster counts
    print("Evaluating k = 2 to 6:\n")
    results = []
    
    for num_clusters in range(2, 7):
        print(f"  Testing k = {num_clusters}...")
        try:
            output_file = os.path.join(
                OUTPUT_DIR, 
                f"{os.path.splitext(os.path.basename(input_path))[0]}_k{num_clusters}.csv"
            )
            
            clustering_result = run_clustering(
                input_path=input_path,
                feature_cols=selected_features,
                algorithm="kmeans",
                k=num_clusters,
                standardise=True,
                output_path=output_file,
                random_state=42,
            )
            
            metrics_dict = clustering_result.get("metrics", {})
            sil_score = metrics_dict.get("silhouette", 0)
            inertia_val = metrics_dict.get("inertia", 0)
            
            results.append({
                'k': num_clusters,
                'silhouette': sil_score,
                'inertia': inertia_val,
            })
            
            print(f"    Silhouette: {sil_score:.4f}, Inertia: {inertia_val:.2f}")
            
        except Exception as e:
            print(f"    Failed: {e}")
            continue

    # Compile results
    results_df = pd.DataFrame(results)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    results_csv_path = os.path.join(OUTPUT_DIR, f"{base_name}_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    print(f"\n{'='*50}")
    print("Results Summary:")
    print(f"{'='*50}")
    print(results_df.to_string(index=False))
    print(f"Saved to: {results_csv_path}\n")

    # Find best clustering
    if not results_df.empty:
        best_idx = results_df['silhouette'].idxmax()
        best_k = int(results_df.loc[best_idx, 'k'])
        best_silhouette = results_df.loc[best_idx, 'silhouette']
        
        print(f"{'='*50}")
        print(f"Best clustering: k = {best_k} (silhouette: {best_silhouette:.4f})")
        print(f"{'='*50}\n")
        
        # Create comparison visualization (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Silhouette comparison
        axes[0, 0].plot(results_df['k'], results_df['silhouette'], 
                    marker='o', linewidth=2.5, markersize=10, color='steelblue')
        axes[0, 0].axvline(x=best_k, color='red', linestyle='--', linewidth=2, 
                       label=f'Best: k={best_k}')
        axes[0, 0].fill_between(results_df['k'], results_df['silhouette'], 
                            alpha=0.25, color='steelblue')
        axes[0, 0].set_xlabel('k (number of clusters)', fontsize=10, fontweight='bold')
        axes[0, 0].set_ylabel('Silhouette score', fontsize=10, fontweight='bold')
        axes[0, 0].set_title('Silhouette Score vs Cluster Count\n(higher is better)', 
                         fontsize=11, fontweight='bold')
        axes[0, 0].set_xticks(results_df['k'])
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Inertia comparison (elbow)
        axes[0, 1].plot(results_df['k'], results_df['inertia'], 
                    marker='s', linewidth=2.5, markersize=10, color='darkviolet')
        axes[0, 1].axvline(x=best_k, color='red', linestyle='--', linewidth=2, 
                       label=f'Best: k={best_k}')
        axes[0, 1].fill_between(results_df['k'], results_df['inertia'], 
                            alpha=0.25, color='darkviolet')
        axes[0, 1].set_xlabel('k (number of clusters)', fontsize=10, fontweight='bold')
        axes[0, 1].set_ylabel('Inertia', fontsize=10, fontweight='bold')
        axes[0, 1].set_title('Elbow Plot: Inertia vs Cluster Count\n(look for elbow)', 
                         fontsize=11, fontweight='bold')
        axes[0, 1].set_xticks(results_df['k'])
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Silhouette normalized (0-1 scale for comparison)
        sil_norm = (results_df['silhouette'] - results_df['silhouette'].min()) / (results_df['silhouette'].max() - results_df['silhouette'].min())
        axes[1, 0].bar(results_df['k'], sil_norm, color='teal', alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[1, 0].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label=f'Best: k={best_k}')
        axes[1, 0].set_xlabel('k (number of clusters)', fontsize=10, fontweight='bold')
        axes[1, 0].set_ylabel('Normalized silhouette', fontsize=10, fontweight='bold')
        axes[1, 0].set_title('Normalized Silhouette Scores\n(all weighted equally)', 
                         fontsize=11, fontweight='bold')
        axes[1, 0].set_xticks(results_df['k'])
        axes[1, 0].set_ylim([0, 1.2])
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].legend()
        
        # Plot 4: Rate of change in inertia
        inertia_diff = results_df['inertia'].diff().dropna()
        k_diff = results_df['k'].iloc[1:].values
        axes[1, 1].plot(k_diff, -inertia_diff, marker='^', linewidth=2.5, markersize=10, 
                       color='darkorange', label='Inertia reduction')
        axes[1, 1].axvline(x=best_k, color='red', linestyle='--', linewidth=2, 
                       label=f'Best: k={best_k}')
        axes[1, 1].fill_between(k_diff, -inertia_diff, alpha=0.25, color='darkorange')
        axes[1, 1].set_xlabel('k (number of clusters)', fontsize=10, fontweight='bold')
        axes[1, 1].set_ylabel('Inertia reduction', fontsize=10, fontweight='bold')
        axes[1, 1].set_title('Rate of Change in Inertia\n(diminishing returns)', 
                         fontsize=11, fontweight='bold')
        axes[1, 1].set_xticks(k_diff)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        comparison_plot = os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png")
        plt.savefig(comparison_plot, dpi=120, bbox_inches='tight')
        print(f"Saved comparison plot: {comparison_plot}\n")
        plt.close()
        
        # Run final clustering with best k (but don't save CSV)
        print(f"Computing final clustering with k = {best_k}...\n")
        try:
            final_clustering = run_clustering(
                input_path=input_path,
                feature_cols=selected_features,
                algorithm="kmeans",
                k=best_k,
                standardise=True,
                output_path=None,
                random_state=42,
            )
            
            final_labels = final_clustering.get("labels", None)
            final_X = final_clustering.get("X_standardised", None)
            
            # Detailed analysis if data available
            if final_X is not None and final_labels is not None:
                print("Detailed cluster analysis:")
                analysis = silhouette_score_sklearn(final_X, final_labels)
                print(f"  Mean silhouette: {analysis['silhouette_mean']:.4f}")
                print(f"  Assessment: {analysis['quality_assessment']}")
                print(f"\n  Per-cluster scores:")
                for cid in sorted(analysis['silhouette_by_cluster'].keys()):
                    score = analysis['silhouette_by_cluster'][cid]
                    print(f"    Cluster {cid}: {score:.4f}")
            
            print(f"\nFinal clustering complete")
            
            # Summary
            print(f"\n{'='*50}")
            print("ANALYSIS COMPLETE")
            print(f"{'='*50}")
            print(f"Optimal clusters: {best_k}")
            print(f"Quality score: {best_silhouette:.4f}")
            print(f"\nOutputs generated:")
            print(f"  - {results_csv_path}")
            print(f"  - {comparison_plot}")
            print(f"{'='*50}\n")
            
        except Exception as e:
            print(f"Error in final clustering: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])