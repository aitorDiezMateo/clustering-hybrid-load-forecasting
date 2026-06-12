import pandas as pd
import os
import sys
import datetime
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from functions.clustering_intervals import assign_cluster_to_profiles


def run(config: dict) -> None:
    cfg = config.get("interval_clustering", {})
    CREATE_ARI_NMI_HEATMAP = cfg.get("create_ari_nmi_heatmap", False)
    OPTIMAL_K_VALUES = cfg.get("optimal_k_values", [2, 3])
    CHOSEN_INTERVAL_CONFIGURATION_INDEX = cfg.get("chosen_interval_configuration_index", 1)
    USE_FILTER = cfg.get("use_filter", True)

    root_dir = config.get("_project_root", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root_dir, "data")
    clustered_dir = os.path.join(root_dir, "output", "clustered_data")

    os.makedirs(clustered_dir, exist_ok=True)

    df_energy = pd.read_csv(
        os.path.join(data_dir, "data.csv"),
        parse_dates=["Date"], usecols=["Date", "total", "temperature", "radiation", "total_filter"]
    )
    df_energy.set_index("Date", inplace=True)

    if USE_FILTER:
        df_energy["total"] = df_energy["total_filter"]

    col = df_energy["total"]
    df_energy["total_normalized"] = (col - col.min()) / (col.max() - col.min())

    df_plot = df_energy[["total_normalized"]].copy()
    df_plot["DateOnly"] = df_plot.index.date
    df_plot["TimeOfDay"] = df_plot.index.time

    pivot_df = df_plot.pivot(index="DateOnly", columns="TimeOfDay", values="total_normalized")
    df_time_base = pivot_df.reindex(columns=sorted(pivot_df.columns)).dropna(how='all')

    def str_to_time(s: str) -> datetime.time:
        return datetime.datetime.strptime(s, "%H:%M").time()

    list_intervals_config = [
        {
        "I1": (str_to_time("01:00"), str_to_time("06:00")),
        "I2": (str_to_time("06:00"), str_to_time("08:00")),
        "I3": (str_to_time("08:00"), str_to_time("12:00")),
        "I4": (str_to_time("12:00"), str_to_time("15:00")),
        "I5": (str_to_time("15:00"), str_to_time("18:00")),
        "I6": (str_to_time("18:00"), str_to_time("21:30")),
        "I7": (str_to_time("21:00"), str_to_time("01:00")),
        },
        {
        "I1": (str_to_time("01:00"), str_to_time("06:00")),
        "I2": (str_to_time("06:00"), str_to_time("07:00")),
        "I3": (str_to_time("07:00"), str_to_time("08:00")),
        "I4": (str_to_time("08:00"), str_to_time("09:00")),
        "I5": (str_to_time("09:00"), str_to_time("10:00")),
        "I6": (str_to_time("10:00"), str_to_time("12:00")),
        "I7": (str_to_time("12:00"), str_to_time("15:00")),
        "I8": (str_to_time("15:00"), str_to_time("18:00")),
        "I9": (str_to_time("18:00"), str_to_time("21:30")),
        "I10": (str_to_time("21:00"), str_to_time("01:00")),
        },
        {
        "I1": (str_to_time("01:00"), str_to_time("06:00")),
        "I2": (str_to_time("06:00"), str_to_time("08:00")),
        "I3": (str_to_time("08:00"), str_to_time("10:00")),
        "I4": (str_to_time("10:00"), str_to_time("12:00")),
        "I5": (str_to_time("12:00"), str_to_time("15:00")),
        "I6": (str_to_time("15:00"), str_to_time("18:00")),
        "I7": (str_to_time("18:00"), str_to_time("21:30")),
        "I8": (str_to_time("21:00"), str_to_time("01:00")),
        },
        {
        "I1": (str_to_time("01:00"), str_to_time("06:00")),
        "I2": (str_to_time("06:00"), str_to_time("08:00")),
        "I3": (str_to_time("08:00"), str_to_time("12:00")),
        "I4": (str_to_time("12:00"), str_to_time("14:00")),
        "I5": (str_to_time("14:00"), str_to_time("15:00")),
        "I6": (str_to_time("15:00"), str_to_time("18:00")),
        "I7": (str_to_time("18:00"), str_to_time("21:30")),
        "I8": (str_to_time("21:00"), str_to_time("01:00")),
        },
    ]

    df_clusters = pd.DataFrame(index=df_time_base.index)
    counter = 0

    for intervals in list_intervals_config:
        counter += 1

        df_intervals = pd.DataFrame(index=df_time_base.index)

        for name, (start, end) in intervals.items():
            if start < end:
                cols = [col for col in df_time_base.columns if start <= col <= end]
            else:
                cols = [col for col in df_time_base.columns if col >= start or col <= end]
            df_intervals[name] = df_time_base[cols].mean(axis=1)

        for optimal_k in OPTIMAL_K_VALUES:
            df_intervals_clustered, df_time_clustered = assign_cluster_to_profiles(
                df_intervals, df_time_base, k=optimal_k, visualize=False
            )

            df_clusters[f"cluster_{counter}_k_{optimal_k}"] = df_time_clustered["cluster"]

            df_intervals_clustered.to_csv(
                os.path.join(clustered_dir, f"df_intervals_clustered_filter_{USE_FILTER}_k_{optimal_k}_config_{counter}.csv")
            )
            df_time_clustered.to_csv(
                os.path.join(clustered_dir, f"df_time_clustered_filter_{USE_FILTER}_k_{optimal_k}_config_{counter}.csv")
            )

            cluster_mean_profiles = df_time_clustered.groupby('cluster').mean()
            print(f"Mean profiles shape: {cluster_mean_profiles.shape}")
            print(f"Number of clusters: {len(cluster_mean_profiles)}")

            cluster_mean_profiles.to_csv(
                os.path.join(clustered_dir, f"cluster_mean_profiles_filter_{USE_FILTER}_k_{optimal_k}_config_{counter}.csv")
            )

    if CREATE_ARI_NMI_HEATMAP:
        config_names = [f"Config_{i+1}" for i in range(len(list_intervals_config))]
        ari_matrix = pd.DataFrame(index=config_names, columns=config_names, dtype=float)
        nmi_matrix = pd.DataFrame(index=config_names, columns=config_names, dtype=float)

        for i, name_i in enumerate(config_names):
            for j, name_j in enumerate(config_names):
                labels_i = df_clusters[f"cluster_{i+1}_k_{OPTIMAL_K_VALUES[-1]}"]
                labels_j = df_clusters[f"cluster_{j+1}_k_{OPTIMAL_K_VALUES[-1]}"]
                ari_matrix.loc[name_i, name_j] = adjusted_rand_score(labels_i, labels_j)
                nmi_matrix.loc[name_i, name_j] = normalized_mutual_info_score(labels_i, labels_j)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(ari_matrix, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True, ax=axes[0],
                    linewidths=0.5)
        axes[0].set_title("Adjusted Rand Index (ARI)", fontsize=14)
        axes[0].tick_params(axis="x", rotation=45)
        sns.heatmap(nmi_matrix, annot=True, cmap="PuBuGn", fmt=".2f", cbar=True, ax=axes[1],
                    linewidths=0.5)
        axes[1].set_title("Normalized Mutual Information (NMI)", fontsize=14)
        axes[1].tick_params(axis="x", rotation=45)
        plt.tight_layout()
        heatmap_dir = os.path.join(root_dir, "output", "intervals_configurations")
        os.makedirs(heatmap_dir, exist_ok=True)
        plt.savefig(os.path.join(
            heatmap_dir,
            f"clustering_comparison_heatmaps_filter_{USE_FILTER}_k_{OPTIMAL_K_VALUES[-1]}_config_{CHOSEN_INTERVAL_CONFIGURATION_INDEX}.png"
        ))
        plt.close(fig)

    print("All files have been saved successfully.")


_STANDALONE_CONFIG = {
    "interval_clustering": {
        "optimal_k_values": [2, 3],
        "chosen_interval_configuration_index": 1,
        "use_filter": True,
        "create_ari_nmi_heatmap": False,
    }
}

if __name__ == "__main__":
    run(_STANDALONE_CONFIG)
