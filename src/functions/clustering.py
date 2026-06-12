from __future__ import annotations

# Set environment variable to prevent KMeans memory leak on Windows
import base64
import io
import os
import sys
from typing import Iterable, Sequence

os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from normalization import pivot_data

def create_report(
    df: pd.DataFrame,
    targets: Sequence[str] | None = None,
    temperature_col: str = "temperature",
    radiation_col: str = "radiation",
    output_dir: str = "output",
    output_file: str = "clustering_report.html",
    k_range: Iterable[int] = range(2, 10),
) -> None:
    """Create an interactive HTML clustering report for daily profiles."""
    
    # If targets is not provided, we will use all the columns
    if targets is None:
        targets = df.columns.difference([temperature_col, radiation_col]).tolist()
    else:
        targets = list(targets)
    
    # Initialize the HTML content
    html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clustering Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    line-height: 1.6;
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 60px auto 0;
                    padding: 20px;
                }}
                .section {{
                    margin-bottom: 40px;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background-color: #fff;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    max-height: 800px;
                    overflow-y: auto;
                }}
                .figure {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .row {{
                    display: flex;
                    justify-content: space-between;
                }}
                .column {{
                    flex: 48%;
                    margin: 0 1%;
                }}
                .cluster-info {{
                    margin: 15px 0;
                    font-weight: bold;
                }}
                .timestamp {{
                    color: #999;
                    font-size: 0.8em;
                    text-align: right;
                    margin-top: 20px;
                }}
                .selector {{
                    margin: 20px 0;
                    padding: 10px;
                    background-color: #eee;
                    border-radius: 5px;
                    text-align: center;
                }}
                .selector select {{
                    padding: 5px;
                    font-size: 16px;
                    border-radius: 4px;
                }}
                .selector button {{
                    padding: 5px 15px;
                    margin-left: 10px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 16px;
                }}
                .selector button:hover {{
                    background-color: #45a049;
                }}
                .cluster-container {{
                    margin-top: 20px;
                }}
                .cluster-option {{
                    display: none;
                }}
                .active {{
                    display: block;
                }}
                .navbar {{
                    position: fixed;
                    top: 0;
                    width: 100%;
                    background-color: #333;
                    overflow: hidden;
                    z-index: 1000;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                }}
                .navbar a {{
                    float: left;
                    display: block;
                    color: white;
                    text-align: center;
                    padding: 14px 16px;
                    text-decoration: none;
                    transition: background-color 0.3s;
                }}
                .navbar a:hover {{
                    background-color: #555;
                }}
                .header {{
                    padding: 20px;
                    background-color: white;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
            </style>
            <script>
                function showClusters(targetId, numClusters) {{
                    // Hide all cluster options for this target
                    const options = document.querySelectorAll("[id^='" + targetId + "-clusters-']");
                    options.forEach(option => {{
                        option.classList.remove('active');
                    }});
                    
                    // Show the selected cluster option
                    const selectedOption = document.getElementById(targetId + "-clusters-" + numClusters);
                    if (selectedOption) {{
                        selectedOption.classList.add('active');
                    }}
                }}
            </script>
        </head>
        <body>
            <div class="navbar">
                <a href="#top">Start</a>
                {' '.join([f'<a href="#{target}">{target}</a>' for target in targets])}
            </div>
            
            <div class="container">
                <div id="top" class="header">
                    <h1>Clustering Analysis Report</h1>
                    <p>This report presents K-means clustering analysis for daily consumption patterns of multiple smart meters. 
                    For each target, you can select different numbers of clusters to analyze how daily consumption profiles 
                    can be segmented into different patterns.</p>
                </div>
        """
    for target in targets:
        pivot_df = pivot_data(df, target, save_data=False)
        X = pivot_df.fillna(0)
        
        # Calculate metrics for different numbers of clusters
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        
        # Define the range of clusters to test
        K_range = list(k_range)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            
            # Calculate metrics:
            
            # Inertia:
            inertias.append(kmeans.inertia_)
            
            # Silhouette score:
            silhouette_scores.append(_safe_metric(silhouette_score, X, clusters))
            
            # Davies-Bouldin score (lower is better):
            davies_bouldin_scores.append(_safe_metric(davies_bouldin_score, X, clusters))
            
            # Calinski-Harabasz score (higher is better):
            calinski_harabasz_scores.append(_safe_metric(calinski_harabasz_score, X, clusters))
        
        # Create elbow method plot
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, inertias, marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Inertia')
        plt.title(f'Elbow Method for {target}')
        plt.grid(True)
        
        # Save elbow plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        elbow_img = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Create silhouette score plot
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, silhouette_scores, marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title(f'Silhouette Score for {target} (Higher is Better)')
        plt.grid(True)
        
        # Save silhouette plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        silhouette_img = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Create Davies-Bouldin index plot
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, davies_bouldin_scores, marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Davies-Bouldin Index')
        plt.title(f'Davies-Bouldin Index for {target} (Lower is Better)')
        plt.grid(True)
        
        # Save Davies-Bouldin plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        davies_bouldin_img = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Create Calinski-Harabasz index plot
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, calinski_harabasz_scores, marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Calinski-Harabasz Index')
        plt.title(f'Calinski-Harabasz Index for {target} (Higher is Better)')
        plt.grid(True)
        
        # Save Calinski-Harabasz plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        calinski_harabasz_img = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Start the target section in HTML
        html_content += f"""
        <div id="{target}" class="section">
            <h2>Clustering Analysis for {target}</h2>
            
            <div class="row">
                <div class="column">
                    <div class="figure">
                        <h3>Elbow Method</h3>
                        <img src="data:image/png;base64,{elbow_img}" alt="Elbow Method" style="max-width:100%;">
                    </div>
                </div>
                <div class="column">
                    <div class="figure">
                        <h3>Silhouette Score</h3>
                        <img src="data:image/png;base64,{silhouette_img}" alt="Silhouette Score" style="max-width:100%;">
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="column">
                    <div class="figure">
                        <h3>Davies-Bouldin Index</h3>
                        <img src="data:image/png;base64,{davies_bouldin_img}" alt="Davies-Bouldin Index" style="max-width:100%;">
                    </div>
                </div>
                <div class="column">
                    <div class="figure">
                        <h3>Calinski-Harabasz Index</h3>
                        <img src="data:image/png;base64,{calinski_harabasz_img}" alt="Calinski-Harabasz Index" style="max-width:100%;">
                    </div>
                </div>
            </div>
            
            <div class="selector">
                <label for="{target}-cluster-select">Select number of clusters: </label>
                <select id="{target}-cluster-select">
                    <option value="2">2 clusters</option>
                    <option value="3" selected>3 clusters</option>
                    <option value="4">4 clusters</option>
                    <option value="5">5 clusters</option>
                    <option value="6">6 clusters</option>
                    <option value="7">7 clusters</option>
                    <option value="8">8 clusters</option>
                    <option value="9">9 clusters</option>
                    <option value="10">10 clusters</option>
                    <option value="11">11 clusters</option>
                    <option value="12">12 clusters</option>
                </select>
                <button onclick="showClusters('{target}', document.getElementById('{target}-cluster-select').value)">Apply</button>
            </div>
            
            <div class="cluster-container">
        """
        
        # Generate plots for each possible cluster count (2-10)
        for num_clusters in k_range:
            cluster_img = None
            cluster_counts = []
            
            # Perform K-means clustering with selected k
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            
            # Add cluster labels to the pivot dataframe (temporary)
            temp_df = pivot_df.copy()
            temp_df['cluster'] = clusters
            
            # Create cluster plots
            plt.figure(figsize=(12, 4 * num_clusters))
            
            for cluster_id in range(num_clusters):
                plt.subplot(num_clusters, 1, cluster_id + 1)
                
                # Get days belonging to this cluster
                cluster_data = temp_df[temp_df['cluster'] == cluster_id].drop('cluster', axis=1)
                cluster_counts.append(len(cluster_data))
                
                # Plot each day in this cluster
                for idx, (date, row) in enumerate(cluster_data.iterrows()):
                    plt.plot(row.index, row.values, alpha=0.3, label=date if idx == 0 else None, color='#c690d1')
                
                # Calculate and plot the average profile for this cluster
                cluster_mean = cluster_data.mean()
                plt.plot(cluster_mean.index, cluster_mean.values, color='red', linewidth=3, label='Cluster Mean')
                
                plt.title(f'Cluster {cluster_id} ({len(cluster_data)} days)')
                plt.xlabel('Hour of Day')
                plt.ylabel(f"{target} (normalized)")
                plt.grid(True)
                plt.legend()
            
            plt.tight_layout()
            
            # Save clusters plot to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close()
            buffer.seek(0)
            cluster_img = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Add cluster option to HTML
            active_class = "active" if num_clusters == 3 else ""
            html_content += f"""
            <div id="{target}-clusters-{num_clusters}" class="cluster-option {active_class}">
                <div class="figure">
                    <h3>Cluster Profiles ({num_clusters} clusters)</h3>
                    <img src="data:image/png;base64,{cluster_img}" alt="Cluster Profiles" style="max-width:100%;">
                </div>
                
                <div class="cluster-info">
                    <h3>Cluster Distribution:</h3>
            """
            
            for cluster_id in range(num_clusters):
                html_content += f"<p>Cluster {cluster_id}: {cluster_counts[cluster_id]} days</p>"
            
            html_content += """
                </div>
            </div>
            """
        
        # Close the cluster-container and section divs
        html_content += """
            </div>
        </div>
        """
    
    # Close the HTML content
    html_content += """
        <div class="timestamp">
            <p>Analysis completed successfully.</p>
        </div>
    </div>
    </body>
    </html>
    """
    
    # Write the HTML to a file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), 'w') as f:
        f.write(html_content)

def create_report_flexible(
    df: pd.DataFrame,
    targets: Sequence[str] | None = None,
    output_dir: str = "output",
    output_file: str = "clustering_report_flexible.html",
    k_range: Iterable[int] = range(2, 10),
    time_intervals_per_day: int = 96,
    highlight_weekends: bool = False,
) -> None:
    """
    Create a clustering report for any time frequency with already normalized columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index and normalized target columns
    targets : list, optional
        List of target column names to analyze. If None, uses all numeric columns.
    output_dir : str
        Directory to save the output HTML file
    output_file : str
        Name of the output HTML file
    k_range : range
        Range of cluster numbers to test
    time_intervals_per_day : int
        Number of time intervals per day (e.g., 96 for 15-min intervals, 24 for hourly)
    highlight_weekends : bool, optional
        If True, weekend days will be plotted in green color. Default is False.
    """
    
    # If targets is not provided, we will use all numeric columns
    if targets is None:
        targets = df.select_dtypes(include=['number']).columns.tolist()
    else:
        targets = list(targets)
    
    # Initialize the HTML content
    html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clustering Analysis Report (Flexible Frequency)</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                    line-height: 1.6;
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 60px auto 0;
                    padding: 20px;
                }}
                .section {{
                    margin-bottom: 40px;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background-color: #fff;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    max-height: 800px;
                    overflow-y: auto;
                }}
                .figure {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .row {{
                    display: flex;
                    justify-content: space-between;
                }}
                .column {{
                    flex: 48%;
                    margin: 0 1%;
                }}
                .cluster-info {{
                    margin: 15px 0;
                    font-weight: bold;
                }}
                .timestamp {{
                    color: #999;
                    font-size: 0.8em;
                    text-align: right;
                    margin-top: 20px;
                }}
                .selector {{
                    margin: 20px 0;
                    padding: 10px;
                    background-color: #eee;
                    border-radius: 5px;
                    text-align: center;
                }}
                .selector select {{
                    padding: 5px;
                    font-size: 16px;
                    border-radius: 4px;
                }}
                .selector button {{
                    padding: 5px 15px;
                    margin-left: 10px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 16px;
                }}
                .selector button:hover {{
                    background-color: #45a049;
                }}
                .cluster-container {{
                    margin-top: 20px;
                }}
                .cluster-option {{
                    display: none;
                }}
                .active {{
                    display: block;
                }}
                .navbar {{
                    position: fixed;
                    top: 0;
                    width: 100%;
                    background-color: #333;
                    overflow: hidden;
                    z-index: 1000;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                }}
                .navbar a {{
                    float: left;
                    display: block;
                    color: white;
                    text-align: center;
                    padding: 14px 16px;
                    text-decoration: none;
                    transition: background-color 0.3s;
                }}
                .navbar a:hover {{
                    background-color: #555;
                }}
                .header {{
                    padding: 20px;
                    background-color: white;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
            </style>
            <script>
                function showClusters(targetId, numClusters) {{
                    // Hide all cluster options for this target
                    const options = document.querySelectorAll("[id^='" + targetId + "-clusters-']");
                    options.forEach(option => {{
                        option.classList.remove('active');
                    }});
                    
                    // Show the selected cluster option
                    const selectedOption = document.getElementById(targetId + "-clusters-" + numClusters);
                    if (selectedOption) {{
                        selectedOption.classList.add('active');
                    }}
                }}
            </script>
        </head>
        <body>
            <div class="navbar">
                <a href="#top">Start</a>
                {' '.join([f'<a href="#{target}">{target}</a>' for target in targets])}
            </div>
            
            <div class="container">
                <div id="top" class="header">
                    <h1>Clustering Analysis Report (Flexible Frequency)</h1>
                    <p>This report presents K-means clustering analysis for daily consumption patterns. 
                    The analysis uses {time_intervals_per_day} time intervals per day.
                    For each target, you can select different numbers of clusters to analyze how daily consumption profiles 
                    can be segmented into different patterns.</p>
                </div>
        """
    
    for target in targets:
        # Create pivot data manually for flexible frequency
        df_copy = df[[target]].copy()
        # Use a non-ambiguous column name different from any index level name
        df_copy['DateOnly'] = pd.to_datetime(df_copy.index).date
        # Build a per-day running index to represent time slots within the day
        df_copy['TimeIndex'] = df_copy.groupby('DateOnly').cumcount()
        
        # Pivot the data to have one row per day and one column per time slot
        pivot_df = df_copy.pivot(index='DateOnly', columns='TimeIndex', values=target)
        
        # Drop days with missing data
        pivot_df = pivot_df.dropna(how='all')
        X = pivot_df.fillna(0)
        
        # Calculate metrics for different numbers of clusters
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        
        # Define the range of clusters to test
        K_range = list(k_range)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            
            # Calculate metrics:
            
            # Inertia:
            inertias.append(kmeans.inertia_)
            
            # Silhouette score:
            silhouette_scores.append(_safe_metric(silhouette_score, X, clusters))
            
            # Davies-Bouldin score (lower is better):
            davies_bouldin_scores.append(_safe_metric(davies_bouldin_score, X, clusters))
            
            # Calinski-Harabasz score (higher is better):
            calinski_harabasz_scores.append(_safe_metric(calinski_harabasz_score, X, clusters))
        
        # Create elbow method plot
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, inertias, marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Inertia')
        plt.title(f'Elbow Method for {target}')
        plt.grid(True)
        
        # Save elbow plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        elbow_img = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Create silhouette score plot
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, silhouette_scores, marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title(f'Silhouette Score for {target} (Higher is Better)')
        plt.grid(True)
        
        # Save silhouette plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        silhouette_img = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Create Davies-Bouldin index plot
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, davies_bouldin_scores, marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Davies-Bouldin Index')
        plt.title(f'Davies-Bouldin Index for {target} (Lower is Better)')
        plt.grid(True)
        
        # Save Davies-Bouldin plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        davies_bouldin_img = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Create Calinski-Harabasz index plot
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, calinski_harabasz_scores, marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Calinski-Harabasz Index')
        plt.title(f'Calinski-Harabasz Index for {target} (Higher is Better)')
        plt.grid(True)
        
        # Save Calinski-Harabasz plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        calinski_harabasz_img = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Start the target section in HTML
        html_content += f"""
        <div id="{target}" class="section">
            <h2>Clustering Analysis for {target}</h2>
            
            <div class="row">
                <div class="column">
                    <div class="figure">
                        <h3>Elbow Method</h3>
                        <img src="data:image/png;base64,{elbow_img}" alt="Elbow Method" style="max-width:100%;">
                    </div>
                </div>
                <div class="column">
                    <div class="figure">
                        <h3>Silhouette Score</h3>
                        <img src="data:image/png;base64,{silhouette_img}" alt="Silhouette Score" style="max-width:100%;">
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="column">
                    <div class="figure">
                        <h3>Davies-Bouldin Index</h3>
                        <img src="data:image/png;base64,{davies_bouldin_img}" alt="Davies-Bouldin Index" style="max-width:100%;">
                    </div>
                </div>
                <div class="column">
                    <div class="figure">
                        <h3>Calinski-Harabasz Index</h3>
                        <img src="data:image/png;base64,{calinski_harabasz_img}" alt="Calinski-Harabasz Index" style="max-width:100%;">
                    </div>
                </div>
            </div>
            
            <div class="selector">
                <label for="{target}-cluster-select">Select number of clusters: </label>
                <select id="{target}-cluster-select">
                    <option value="2">2 clusters</option>
                    <option value="3" selected>3 clusters</option>
                    <option value="4">4 clusters</option>
                    <option value="5">5 clusters</option>
                    <option value="6">6 clusters</option>
                    <option value="7">7 clusters</option>
                    <option value="8">8 clusters</option>
                    <option value="9">9 clusters</option>
                    <option value="10">10 clusters</option>
                    <option value="11">11 clusters</option>
                    <option value="12">12 clusters</option>
                </select>
                <button onclick="showClusters('{target}', document.getElementById('{target}-cluster-select').value)">Apply</button>
            </div>
            
            <div class="cluster-container">
        """
        
        # Generate plots for each possible cluster count
        for num_clusters in k_range:
            cluster_img = None
            cluster_counts = []
            
            # Perform K-means clustering with selected k
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            
            # Add cluster labels to the pivot dataframe (temporary)
            temp_df = pivot_df.copy()
            temp_df['cluster'] = clusters
            
            # Create cluster plots
            plt.figure(figsize=(12, 4 * num_clusters))
            
            for cluster_id in range(num_clusters):
                plt.subplot(num_clusters, 1, cluster_id + 1)
                
                # Get days belonging to this cluster
                cluster_data = temp_df[temp_df['cluster'] == cluster_id].drop('cluster', axis=1)
                cluster_counts.append(len(cluster_data))
                
                # Plot each day in this cluster
                for idx, (date, row) in enumerate(cluster_data.iterrows()):
                    # Determine color based on weekend status if highlight_weekends is enabled
                    if highlight_weekends:
                        is_weekend = pd.Timestamp(date).weekday() >= 5
                        color = "#4caf50" if is_weekend else "#c690d1"
                    else:
                        color = '#c690d1'
                    
                    plt.plot(row.index, row.values, alpha=0.3, label=date if idx == 0 else None, color=color)
                
                # Calculate and plot the average profile for this cluster
                cluster_mean = cluster_data.mean()
                plt.plot(cluster_mean.index, cluster_mean.values, color='red', linewidth=3, label='Cluster Mean')
                
                plt.title(f'Cluster {cluster_id} ({len(cluster_data)} days)')
                plt.xlabel('Time Interval')
                plt.ylabel(f"{target}")
                plt.grid(True)
                plt.legend()
            
            plt.tight_layout()
            
            # Save clusters plot to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close()
            buffer.seek(0)
            cluster_img = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Add cluster option to HTML
            active_class = "active" if num_clusters == 3 else ""
            html_content += f"""
            <div id="{target}-clusters-{num_clusters}" class="cluster-option {active_class}">
                <div class="figure">
                    <h3>Cluster Profiles ({num_clusters} clusters)</h3>
                    <img src="data:image/png;base64,{cluster_img}" alt="Cluster Profiles" style="max-width:100%;">
                </div>
                
                <div class="cluster-info">
                    <h3>Cluster Distribution:</h3>
            """
            
            for cluster_id in range(num_clusters):
                html_content += f"<p>Cluster {cluster_id}: {cluster_counts[cluster_id]} days</p>"
            
            html_content += """
                </div>
            </div>
            """
        
        # Close the cluster-container and section divs
        html_content += """
            </div>
        </div>
        """
    
    # Close the HTML content
    html_content += """
        <div class="timestamp">
            <p>Analysis completed successfully.</p>
        </div>
    </div>
    </body>
    </html>
    """
    
    # Write the HTML to a file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), 'w') as f:
        f.write(html_content)


def _safe_metric(metric_func, X: pd.DataFrame, clusters: Sequence[int]) -> float:
    try:
        return float(metric_func(X, clusters))
    except Exception:
        return 0.0
