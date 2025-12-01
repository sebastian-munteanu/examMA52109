## What was wrong with the original script

## 1: K value not incrementing correctly (Line 62)
**Original code:**
k = min(k, 3)

This line capped the number of clusters to k ≤ 3, forcing the clustering loop to run with k = 2, 3, 3, 3 instead of the intended k = 2, 3, 4, 5.

**Fix:** 
Changed to k=k, to pass the loop variable directly.



## 2: Silhouette plot not generating due to  (Line 89 and 91)
**Original code:**
if "silhouette_score" in metrics_df.columns:
plt.bar(metrics_df["k"], metrics_df["silhouette_score"])

The actual column name returned by run_clustering() is "silhouette", not "silhouette_score". This caused the silhouette plot to never be generated.

**Fix:** Changed to:
if "silhouette" in metrics_df.columns:
plt.bar(metrics_df["k"], metrics_df["silhouette"])



## 3: Misleading user message (Line 28)
**Original code:**
print("Usage: python clustering_demo.py <input_csv>")

The script is called `cluster_plot.py`, not `clustering_demo.py`, users following the usage message would run the wrong command and encounter errors

**Fix:** 
print("Usage: python demo/cluster_plot.py <input_csv>")

-----------------------------------------------------------------------------

## What the corrected script now does

The script evaluates k-means clustering performance across multiple k values (2, 3, 4, 5) on a 2D dataset and outputs metrics(inertia, silhouette) and plots(cluster plots and silhouette score visualisation), with the following workflow:

1. Validating command-line arguments and inputting CSV data file
2. Extracting the first two numeric columns as features
3. For each k values of 2,3,4 and 5:
   - Running k-means clustering with standardised features
   - Creating and saving clustered data to CSV files
   - Plotting and saving 2D cluster and centroid visualization to PNG files 
   - Outputting clustering metrics (silhouette score, inertia) in the terminal
4. Creatting and saving a summary metrics table to a CSV file
5. Plotting and saving the silhouette scores across k values as a bar chart

-------------------------------------------------------------------------------


## Overview of cluster_maker package

cluster_maker is a comprehensive Python package for unsupervised clustering analysis with integrated preprocessing, visualization, and quality assessment.

### Main Components

1. algorithms.py
- Provides clustering algorithms to group data points into k clusters.

- Implements K-means clustering in two ways:
    - Manual implementation: initialises random centroids, then iteratively assigns points to the nearest centroid and updates centroid positions until convergence.
    - Scikit-learn wrapper: uses the optimised scikit-learn implementation for better performance.
Both return cluster labels and final centroid positions.



2. preprocessing.py
- Prepares raw data for clustering by handling data quality and normalisation.
- Validates that requested feature columns exist and are numeric, then standardises features using z-score normalisation (zero mean, unit variance). This ensures all features contribute equally to distance calculations, regardless of their original scale.


3. dataframe_builder.py
- Generates synthetic clustered data for testing and validation.
- Takes cluster centre coordinates and generates random points around each centre using Gaussian noise. Returns a DataFrame with ground truth cluster labels, allowing algorithm correctness to be verified.

4. evaluation.py
- Computes quantitative metrics to assess clustering quality.
- Calculates multiple metrics including:
    - Silhouette score: measures how well each point fits its cluster (-1 to 1, higher is better)
    - Inertia: sum of squared distances within clusters (lower is better for elbow method)
    - Calinski-Harabasz and Davies-Bouldin indices: measure cluster separation and compactness
These metrics help identify optimal k values and assess whether clusters are meaningful.

5. data_analyser.py
- Computes descriptive statistics for exploratory data analysis.
- Examines numeric columns in a DataFrame and returns mean, standard deviation, min, max, and missing value count for each. Automatically ignores non-numeric columns. Provides a quick overview of data scale and quality before clustering.

6. data_exporter.py
- Writes analysis results to disk in both machine and human-readable formats.
- Exports summary statistics to CSV (for further processing) and to formatted text files (for human reading). Validates output directories to prevent silent failures.

7. plotting_clustered.py
- Creates visual representations of clustering results.
- Generates 2D scatter plots showing data points coloured by cluster and centroid positions marked distinctly. Also produces elbow plots (inertia vs. k) to help identify optimal cluster counts. Outputs high-quality PNG images.

8. interface.py
- Provides a single high-level entry point that coordinates all components.
- The `run_clustering()` function orchestrates the complete workflow: loads data, validates and standardises features, applies clustering, computes metrics, generates visualisations, and exports results. Users call one function instead of managing multiple components separately.

## Workflow

- Input CSV 
- Load & validate data
- Select numeric features
- Standardise features (z-score)
- Apply k-means algorithm
- Compute metrics (silhouette, inertia, Davies-Bouldin, etc.)
- Visualise clusters (2D scatter plots)
- Export results (CSV + PNG plots)
