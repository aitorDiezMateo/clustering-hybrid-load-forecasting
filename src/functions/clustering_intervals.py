from __future__ import annotations

# Set environment variable to prevent KMeans memory leak on Windows
import os
from typing import Iterable

os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

def cluster_analysis_intervals(
    df_intervals: pd.DataFrame,
    k_range: Iterable[int] = range(2, 10),
    output_dir: str | None = None,
) -> dict[str, list[float]]:
    """
    Perform clustering analysis on interval-averaged daily profiles.
    
    Parameters:
    -----------
    df_intervals : pd.DataFrame
        DataFrame with daily rows and interval columns (I1, I2, ..., I7)
    k_range : range
        Range of cluster numbers to test
    output_dir : str, optional
        Directory to save plots
    
    Returns:
    --------
    dict : Dictionary containing clustering metrics for each k value
    """
    
    # Fill any NaN values with 0
    X = df_intervals.fillna(0)
    
    # Store metrics
    metrics = {
        'k': [],
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }
    
    print("\n" + "="*70)
    print("CLUSTERING ANALYSIS ON TIME INTERVALS")
    print("="*70)
    print(f"\nData shape: {X.shape[0]} days x {X.shape[1]} intervals")
    print(f"Intervals: {list(X.columns)}")
    
    # Calculate metrics for different k values
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        metrics['k'].append(k)
        metrics['inertia'].append(kmeans.inertia_)

        metrics['silhouette'].append(_safe_metric(silhouette_score, X, clusters))
        metrics['davies_bouldin'].append(_safe_metric(davies_bouldin_score, X, clusters))
        metrics['calinski_harabasz'].append(_safe_metric(calinski_harabasz_score, X, clusters))
    
    # Plot metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Elbow plot
    axes[0, 0].plot(metrics['k'], metrics['inertia'], marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Elbow Method')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Silhouette score
    axes[0, 1].plot(metrics['k'], metrics['silhouette'], marker='o', linewidth=2, color='green')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Score (Higher is Better)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Davies-Bouldin index
    axes[1, 0].plot(metrics['k'], metrics['davies_bouldin'], marker='o', linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Davies-Bouldin Index')
    axes[1, 0].set_title('Davies-Bouldin Index (Lower is Better)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Calinski-Harabasz index
    axes[1, 1].plot(metrics['k'], metrics['calinski_harabasz'], marker='o', linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Calinski-Harabasz Score')
    axes[1, 1].set_title('Calinski-Harabasz Score (Higher is Better)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "clustering_metrics_intervals.png"), dpi=150, bbox_inches='tight')
        print(f"\nMetrics plot saved to: {os.path.join(output_dir, 'clustering_metrics_intervals.png')}")
    plt.close(fig)
    
    # Print metrics summary
    print("\n" + "-"*70)
    print("CLUSTERING METRICS SUMMARY")
    print("-"*70)
    print(f"{'k':<4} {'Inertia':<12} {'Silhouette':<12} {'Davies-Bouldin':<16} {'Calinski-Harabasz':<18}")
    print("-"*70)
    for i, k in enumerate(metrics['k']):
        print(f"{k:<4} {metrics['inertia'][i]:<12.2f} {metrics['silhouette'][i]:<12.4f} "
              f"{metrics['davies_bouldin'][i]:<16.4f} {metrics['calinski_harabasz'][i]:<18.2f}")
    print("-"*70)
    
    # Recommend optimal k based on metrics
    best_silhouette_k = metrics['k'][metrics['silhouette'].index(max(metrics['silhouette']))]
    valid_db = [x for x in metrics['davies_bouldin'] if x > 0]
    best_db_k = metrics['k'][metrics['davies_bouldin'].index(min(valid_db))] if valid_db else None
    
    print("\nRecommendations:")
    print(f"  Best k by Silhouette Score: {best_silhouette_k}")
    if best_db_k is not None:
        print(f"  Best k by Davies-Bouldin Index: {best_db_k}")
    else:
        print("  Best k by Davies-Bouldin Index: N/A")
    
    return metrics

def visualize_clusters(
    df_intervals: pd.DataFrame,
    k: int,
    output_dir: str | None = None,
) -> None:
    """
    Visualize clusters for a specific k value.
    
    Parameters:
    -----------
    df_intervals : pd.DataFrame
        DataFrame with daily rows and interval columns
    k : int
        Number of clusters
    output_dir : str, optional
        Directory to save plots
    """
    
    X = df_intervals.fillna(0)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Add cluster labels
    df_with_clusters = df_intervals.copy()
    df_with_clusters['cluster'] = clusters
    
    # Add weekday information
    df_with_clusters['is_weekend'] = df_with_clusters.index.map(lambda x: pd.Timestamp(x).weekday() >= 5)
    
    # Create visualization
    fig, axes = plt.subplots(k, 1, figsize=(14, 4*k))
    if k == 1:
        axes = [axes]
    
    colors = ['#c690d1', '#4caf50']  # Purple for weekdays, green for weekends
    
    for cluster_id in range(k):
        ax = axes[cluster_id]
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        
        # Count weekdays and weekends
        n_weekdays = (~cluster_data['is_weekend']).sum()
        n_weekends = cluster_data['is_weekend'].sum()
        
        # Plot individual profiles
        for date, row in cluster_data.iterrows():
            color_idx = 1 if row['is_weekend'] else 0
            ax.plot(df_intervals.columns, row[df_intervals.columns], 
                   alpha=0.25, color=colors[color_idx], linewidth=1)
        
        # Plot cluster centroid
        centroid = cluster_data[df_intervals.columns].mean()
        ax.plot(df_intervals.columns, centroid, color='red', linewidth=3, 
               label='Cluster Mean', marker='o', markersize=8)
        
        ax.set_title(f'Cluster {cluster_id} ({len(cluster_data)} days: {n_weekdays} weekdays, {n_weekends} weekends)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Interval', fontsize=11)
        ax.set_ylabel('Normalized Energy', fontsize=11)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Add background colors to distinguish weekdays/weekends if both present
        if n_weekdays > 0 and n_weekends > 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors[0], alpha=0.3, label=f'Weekdays ({n_weekdays})'),
                Patch(facecolor=colors[1], alpha=0.3, label=f'Weekends ({n_weekends})'),
                ax.get_legend().legend_handles[0]
            ]
            ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"clusters_k{k}_intervals.png"), dpi=150, bbox_inches='tight')
        print(f"\nCluster visualization saved to: {os.path.join(output_dir, f'clusters_k{k}_intervals.png')}")
    plt.close(fig)
    
    # Print cluster statistics
    print("\n" + "="*70)
    print(f"CLUSTER ANALYSIS (k={k})")
    print("="*70)
    
    for cluster_id in range(k):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        n_weekdays = (~cluster_data['is_weekend']).sum()
        n_weekends = cluster_data['is_weekend'].sum()
        
        print(f"\nCluster {cluster_id}:")
        print(f"  Total days: {len(cluster_data)}")
        print(f"  Weekdays: {n_weekdays} ({100*n_weekdays/len(cluster_data):.1f}%)")
        print(f"  Weekends: {n_weekends} ({100*n_weekends/len(cluster_data):.1f}%)")
        print("  Centroid values:")
        for interval, value in cluster_data[df_intervals.columns].mean().items():
            print(f"      {interval}: {value:.4f}")
    
    print("="*70)

def assign_cluster_to_profiles(
    df_intervals: pd.DataFrame,
    df_time: pd.DataFrame,
    k: int = 3,
    visualize: bool = True,
    output_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign clusters based on interval data and optionally visualize detailed daily profiles using full time series.
    
    Parameters:
    -----------
    df_intervals : pd.DataFrame
        DataFrame with daily rows and interval columns (I1, I2, ..., I7)
    df_time : pd.DataFrame
        DataFrame with daily rows and all time points as columns
    k : int
        Number of clusters
    visualize : bool
        If True, create and display/save visualizations. If False, only assign clusters.
    output_dir : str, optional
        Directory to save plots (only used if visualize=True)
    
    Returns:
    --------
    tuple : (df_intervals_with_clusters, df_time_with_clusters)
    """
    import datetime
    import matplotlib.dates as mdates
    
    print("\nAssigning clusters to daily profiles...")
    
    # Perform clustering on interval data
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_intervals.fillna(0))
    
    # Add cluster labels to df_intervals
    df_intervals_clustered = df_intervals.copy()
    df_intervals_clustered['cluster'] = clusters
    
    # Add cluster labels to df_time (merge based on date index)
    df_time_clustered = df_time.copy()
    df_time_clustered['cluster'] = clusters
    
    print(f"Assigned {k} clusters to {len(df_intervals)} days")
    print(f"\nCluster distribution:")
    for cluster_id in range(k):
        count = (df_time_clustered['cluster'] == cluster_id).sum()
        print(f"  Cluster {cluster_id}: {count} days")
    
    # Only create visualizations if requested
    if visualize:
        # Plot detailed profiles for each cluster using df_time
        print("\nCreating detailed daily profiles for each cluster...")
        
        # Create a comprehensive plot with all clusters
        fig, axes = plt.subplots(k, 1, figsize=(14, 5 * k))
        if k == 1:
            axes = [axes]
        
        colors = ['#c690d1', '#4caf50']  # Purple for weekdays, green for weekends
        
        for cluster_id in range(k):
            ax = axes[cluster_id]
            
            # Get days in this cluster
            cluster_mask = df_time_clustered['cluster'] == cluster_id
            cluster_data = df_time_clustered[cluster_mask].drop('cluster', axis=1)
            
            # Count weekdays and weekends
            n_weekdays = sum(1 for date in cluster_data.index if pd.Timestamp(date).weekday() < 5)
            n_weekends = len(cluster_data) - n_weekdays
            
            # Plot each day's profile
            for date, row in cluster_data.iterrows():
                is_weekend = pd.Timestamp(date).weekday() >= 5
                color = colors[1] if is_weekend else colors[0]
                
                # Convert time columns to datetime for plotting
                x_times = [datetime.datetime.combine(datetime.datetime(2000, 1, 1).date(), t) 
                           for t in row.dropna().index]
                
                ax.plot(x_times, row.dropna().values, alpha=0.25, color=color, linewidth=1)
            
            # Calculate and plot the mean profile
            cluster_mean = cluster_data.mean()
            x_times_mean = [datetime.datetime.combine(datetime.datetime(2000, 1, 1).date(), t) 
                            for t in cluster_mean.dropna().index]
            ax.plot(x_times_mean, cluster_mean.dropna().values, color='red', 
                    linewidth=3, label='Cluster Mean', marker='o', markersize=4)
            
            # Format plot
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f'Cluster {cluster_id} - Detailed Daily Profiles ({len(cluster_data)} days: {n_weekdays} weekdays, {n_weekends} weekends)', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time of Day', fontsize=11)
            ax.set_ylabel('Normalized Energy Consumption', fontsize=11)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors[0], alpha=0.3, label=f'Weekdays ({n_weekdays})'),
                Patch(facecolor=colors[1], alpha=0.3, label=f'Weekends ({n_weekends})'),
                plt.Line2D([0], [0], color='red', linewidth=3, label='Cluster Mean')
            ]
            ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"detailed_profiles_k{k}.png"), 
                        dpi=150, bbox_inches='tight')
            print(f"Detailed profiles saved to: {os.path.join(output_dir, f'detailed_profiles_k{k}.png')}")
        plt.close(fig)
        
        print("\nDetailed profile visualization completed successfully.")
    
    # Save cluster assignments to CSV if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df_intervals_clustered.to_csv(os.path.join(output_dir, "df_intervals_with_clusters.csv"))
        print(f"Cluster assignments saved to: {os.path.join(output_dir, 'df_intervals_with_clusters.csv')}")
    
    return df_intervals_clustered, df_time_clustered


# Backward compatibility alias
def assign_and_visualize_detailed_profiles(
    df_intervals: pd.DataFrame,
    df_time: pd.DataFrame,
    k: int = 3,
    output_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backward compatibility wrapper for assign_cluster_to_profiles with visualize=True.
    
    This function is deprecated. Please use assign_cluster_to_profiles instead.
    """
    return assign_cluster_to_profiles(df_intervals, df_time, k=k, visualize=True, output_dir=output_dir)




def _safe_metric(metric_func, X: pd.DataFrame, clusters: Iterable[int]) -> float:
    try:
        return float(metric_func(X, clusters))
    except Exception:
        return 0.0


