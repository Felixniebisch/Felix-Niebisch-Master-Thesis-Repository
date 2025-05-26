# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:17:07 2025

@author: Gulraiz.Iqbal and Felixniebisch
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import os
from BehaviorCSVReader import BehaviorCSVReader
from CSVDataReader import CSVDataReader
from datetime import datetime
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from collections import Counter

trajectory_length_cutoff = 2000

def extract_trialwise_dlc_trajectories(beh_csv, dlc_csv, dlc_fps=25, bodypart="Nose", likelihood_thresh=0.99):
    beh_csv = Path(beh_csv)
    br = BehaviorCSVReader(beh_csv.parent, beh_csv.name)
    _, info = br.read_behavior_csv()


    trial_windows = info.get("TrialWindows", [])
    correctness_list = info.get("CorrectTrials", [])
    if not trial_windows or not correctness_list:
        print(f"Missing trial window or correctness in {beh_csv}")

        return [], [], []

    cdr = CSVDataReader(dlc_csv)
    parts = cdr.get_data_from_csv(exclude_list=["bodyparts"], omit_prediction=False)
    if bodypart not in parts:
        print(f"Bodypart '{bodypart}' missing in {dlc_csv}")
        return [], [], []

    arr = parts[bodypart]  # shape: (n_frames, 3) ==> timestamp, x-, y coordinate
    frame_times = np.arange(arr.shape[0]) * (1000.0 / dlc_fps) # upsampling to have millisecond rates for both

    valid_trajs, valid_correctness, valid_sides = [], [], []
    for i, (start_ms, end_ms, side) in enumerate(trial_windows):
        if len(trial_windows) >= trajectory_length_cutoff:
            continue
        mask = (frame_times >= start_ms) & (frame_times <= end_ms) # trajectory length
        coords = arr[mask]
        t = frame_times[mask]

        # Filter by likelihood
        coords = coords[coords[:, 2] > likelihood_thresh] # bade coordinates
        t = t[:coords.shape[0]]
        
        #first_x, first_y = coords[0, 0], coords[0, 1] # uncomment if you want to analyze starting positions of trajectories
        #first_time = t[0]
        
        #print(f"Trial {i}: First coordinate = ({first_x:.1f}, {first_y:.1f}) at t = {(first_time / 1000) * 25:.1f} ms")
        if len(coords) == 0 or coords.shape[0] != len(t):
            continue
        
        if len(t) == 0 or (t[-1] - t[0]) > trajectory_length_cutoff: # cutting off trajectories that are longer than 2000ms
           continue
       
        print(len(t))
        
                
        # Exclude trials not starting within middle sensor region
        center_x, center_y = 600, 600  # position of middle sensor
        start_radius_thresh = 90       
        start_x, start_y = coords[0, 0], coords[0, 1]
        dist_from_center = np.sqrt((start_x - center_x)**2 + (start_y - center_y)**2)
        if dist_from_center > start_radius_thresh:
            continue
       
        xyt = np.column_stack([coords[:, 0], coords[:, 1], t]) # xyz array
        valid_trajs.append(xyt)

        # Add corresponding behavioral info
        if i < len(correctness_list):
            valid_correctness.append(correctness_list[i])
            valid_sides.append(side)
        else:
            valid_correctness.append(None)
            valid_sides.append(None)

    return valid_trajs, valid_correctness, valid_sides # retain correctness and initial side of the trajectories for later analysis 

def interpolate_space_time(traj, M=20):
    traj = np.array(traj)
    
    # Calculate cumulative spatial arc length (x, y only)
    spatial_diffs = np.diff(traj[:, :2], axis=0)
    spatial_dists = np.linalg.norm(spatial_diffs, axis=1)
    arc_lengths = np.insert(np.cumsum(spatial_dists), 0, 0)

    # Uniform arc lengths
    uniform_arc = np.linspace(0, arc_lengths[-1], M)

    # Interpolate x, y, t over uniform arc length
    x_interp = interp1d(arc_lengths, traj[:, 0],  kind='linear')(uniform_arc)
    y_interp = interp1d(arc_lengths, traj[:, 1],  kind='linear')(uniform_arc)
    t_interp = interp1d(arc_lengths, traj[:, 2],  kind='linear')(uniform_arc)

    return np.stack([x_interp, y_interp, t_interp], axis=1)


def kmeans(X, K=4, num_iters=10):
    np.random.seed(0)
    indices = np.random.choice(len(X), K, replace=False)
    centroids = X[indices]

    for _ in range(num_iters):
        # Assign points to the closest centroid
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  #calculation for the distance in between the different clusters ( similarity metric )
        labels = np.argmin(distances, axis=1)

        # Update centroids
        for k in range(K):
            if np.sum(labels == k) > 0:
                centroids[k] = X[labels == k].mean(axis=0)

    return labels, centroids

def normalize_trajectories(trajectories):
    normalized_trajectories = []
    for traj in trajectories:
        min_vals = traj.min(axis=0)
        max_vals = traj.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        norm_traj = (traj - min_vals) / range_vals
        normalized_trajectories.append(norm_traj)
    return normalized_trajectories

def spatiotemporal_variability_per_cluster(interpolated_trajectories, labels, K):

    cluster_vars = []

    for k in range(K):
        cluster_trajs = [t for i, t in enumerate(interpolated_trajectories) if labels[i] == k]
        if len(cluster_trajs) == 0:
            cluster_vars.append(np.nan)
            continue

        cluster_arr = np.array(cluster_trajs)  # shape (n_k, M, 3)
        mean_traj = np.mean(cluster_arr, axis=0)  # (M, 3)

        # Euclidean distance from mean per trial
        distances = [
            np.mean(np.linalg.norm(traj - mean_traj, axis=1)) #/ np.sqrt(len(cluster_trajs)) #<=== SEM
            for traj in cluster_arr
        ]
        cluster_vars.append(np.mean(distances))
        print(cluster_vars)

    return cluster_vars



def summarize_clusters(labels, correctness, sides):
    """
    Create summary DataFrame using aligned labels, correctness, and side info.
    """
    min_len = min(len(labels), len(correctness), len(sides))
    labels = labels[:min_len]
    correctness = correctness[:min_len]
    sides = sides[:min_len]

    print(f"  len(labels):      {len(labels)}")
    print(f"  len(correctness): {len(correctness)}")
    print(f"  len(trial_sides): {len(sides)}")

    labeled_df = pd.DataFrame({
        "Cluster": labels,
        "Correct": correctness,
        "Side": sides
    })

    print("\n Cluster vs Correctness")
    print(pd.crosstab(labeled_df["Cluster"], labeled_df["Correct"]))

    print("\n Cluster vs Side")
    print(pd.crosstab(labeled_df["Cluster"], labeled_df["Side"]))

    return labeled_df


def align_clusters(reference_centroids, target_centroids):
    """
    Align cluster labels of the target to match the reference based on centroid similarity.
    """
    distance_matrix = cdist(reference_centroids, target_centroids)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    mapping = dict(zip(col_ind, row_ind))
    return mapping

def remap_labels(labels, mapping):
    """
    Remap labels based on the given mapping.
    """
    return np.vectorize(mapping.get)(labels)

def plot_cluster_distribution(labels_per_session, k):
    """
    Plot the distribution of cluster labels across sessions.
    """
    counts = np.zeros((len(labels_per_session), k))
    for i, labels in enumerate(labels_per_session):
        c = Counter(labels)
        for label in range(k):
            counts[i, label] = c[label]
   
    for label in range(k):
        plt.plot(counts[:, label], label=f"Cluster {label}")
   
    plt.xlabel("Session")
    plt.ylabel("Trajectory Count")
    plt.title("Cluster Prevalence Over Sessions")
    plt.legend()
    plt.tight_layout()
    plt.show()
#beh_csv_path = "/Volumes/Expansion/sem-data/M1791/M1791-250423-170044/M1791-250423-170044_cleaned.csv"
#dlc_csv_path = "/Volumes/Expansion/sem-data/M1791/M1791-250423-170044/top/M1791-250423-1659-topDLC_resnet50_ObjectInteractionOct12shuffle1_200000_filtered.csv"
base_path = "" # base path of mouse data 

labels_per_session = []
centroids_per_session = []
results = []
for root, dirs, files in os.walk(base_path):
    for file in files:
        if "filtered.csv" in file and not file.startswith("._"):
            try:
                dlc_csv_path = Path(root) / file
                beh_csv_path = next((f for f in Path(root).parent.iterdir() if f.suffix == '.csv' and not f.name.startswith("._")), None)
                if beh_csv_path is None:
                    continue

                print(f"Processing: {file}")
                # Extract trial-based trajectories
                trialwise_trajs, correctness_filtered, sides_filtered = extract_trialwise_dlc_trajectories(beh_csv_path, dlc_csv_path)
                print(f"Valid trajectories extracted: {len(trialwise_trajs)}")
                # Normalize and interpolate
                normalized = normalize_trajectories(trialwise_trajs)
                M = 20
                interpolated = [interpolate_space_time(tr, M=M) for tr in normalized]
                
                # Flatten for clustering
                X = np.stack([tr.flatten() for tr in interpolated])
                
                
                # === 3D Plot ===
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']  # 7 colors
                labels = [f"tr{i+1}" for i in range(len(trialwise_trajs))]
                
                for traj_interp, color, label in zip(trialwise_trajs, colors, labels):
                    ax.plot(traj_interp[:, 0], traj_interp[:, 1], traj_interp[:, 2], 'o-', color=color, label=label)
                
                ax.set_title('3D Trajectories (X, Y, Time)', fontsize=15)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Time')
                ax.legend()
                plt.tight_layout()
                plt.show()
                
                
                M=6
                flattened_trajectories = []
                interpolated_trajectories = []  # Save interpolated ones separately for plotting
                
                for traj in trialwise_trajs:
                    traj[:, 2] -= traj[0, 2]  # Normalize time to start at 0
                    traj_interp = interpolate_space_time(traj, M=M)
                    interpolated_trajectories.append(traj_interp)
                    flattened = traj_interp.flatten()
                    flattened_trajectories.append(flattened)
                
                # Stack into single dataset
                X = np.stack(flattened_trajectories)
                
                # === 3D Plot ===
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']  # 7 colors
                plot_labels = [f"tr{i+1}" for i in range(len(flattened_trajectories))]
                for traj_interp, color, label in zip(interpolated_trajectories, colors, labels):
                    ax.plot(traj_interp[:, 0], traj_interp[:, 1], traj_interp[:, 2], 'o-', color=color, label=label)
                
                ax.set_title('3D Interpolation of Trajectories (X, Y, Time)', fontsize=15)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Time')
                ax.legend()
                plt.tight_layout()
                plt.show()
                
                
                # Run K-means
                labels, centroids = kmeans(X, K=3, num_iters=10)

                # Align clusters after the first session
                if centroids_per_session:
                    ref_centroids = centroids_per_session[0]  # always align to the first
                    mapping = align_clusters(ref_centroids, centroids)
                    labels = remap_labels(labels, mapping)
                    centroids = np.array([centroids[i] for i in sorted(mapping.keys(), key=lambda x: mapping[x])])
            
                centroids_per_session.append(centroids)
                labels_per_session.append(labels)
              
                
                # Print cluster labels -- which trajectorie is part of which cluster?
                print("Cluster labels:", labels)

                centroids_reshape = []
                
                for centroid in centroids:
                   centroids_reshape.append(centroid.reshape(-1,3))
                

                
                # === 3D Plot ===
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                centroid_labels = [f"c{i+1}" for i in range(len(centroids_reshape))]
                for cent, color, label in zip(centroids_reshape, colors, centroid_labels):
                    ax.plot(cent[:, 0], cent[:, 1], cent[:, 2], 'o-', color=color, label=label)
                
                ax.set_title('Centroids (X, Y, Time)', fontsize=15)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Time')
                ax.legend()
                plt.tight_layout()
                plt.show()
                
                spatiotemporal_vars = spatiotemporal_variability_per_cluster(interpolated_trajectories, labels, K=3)
                labeled_df = summarize_clusters(labels, correctness_filtered, sides_filtered)
                
                # Make sure your cluster labels and colors are consistent
                unique_clusters = np.unique(centroid_labels)
                # Use your predefined colors list (7 colors in your code)
                # but only pick as many colors as clusters
                cluster_colors = colors[:len(unique_clusters)]
                
                # 3D plot for trajectories, colored by cluster
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                for traj_idx, traj in enumerate(interpolated_trajectories):
                    cluster_id = labels[traj_idx]  # cluster index of this trajectory
                    color = cluster_colors[cluster_id]
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'o-', color=color)
                
                # Plot centroids with same color palette, for reference
                for cent, color, cluster_id in zip(centroids_reshape, cluster_colors, range(len(centroids_reshape))):
                    ax.plot(cent[:, 0], cent[:, 1], cent[:, 2], 's--', color=color)
                
                ax.set_title('3D Trajectories Colored by Cluster', fontsize=15)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Time')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # legend outside plot
                plt.tight_layout()
                plt.show()
                
                
                # Extract session date (from DLC filename)
                datecode = file[6:12]  # e.g., "250423"
                try:
                    session_date = datetime.strptime(datecode, "%y%m%d").strftime("%d.%m.%Y")
                except ValueError:
                    session_date = datecode  # fallback
                
                # Create DataFrame rows
                for i, var in enumerate(spatiotemporal_vars):
                    n_trials = np.sum(np.array(labels) == i)
                    results.append({
                        "Session": session_date,
                        "Cluster": f"Cluster {i}",
                        "N_Trajectories": n_trials,
                        "Spatiotemporal Variability": var,
                    })

      
                
                # Format cluster labels to match
                labeled_df["Cluster"] = labeled_df["Cluster"].apply(lambda x: f"Cluster {x}")
                
                # === Compute global correctness ===
                br = BehaviorCSVReader(beh_csv_path.parent, beh_csv_path.name)
                _, info = br.read_behavior_csv()
                correct_trials_global = info.get("CorrectTrials", [])
                total_trials_global = len(correct_trials_global)
                total_correct_global = sum(correct_trials_global)
                global_pct_correct = total_correct_global / total_trials_global if total_trials_global > 0 else np.nan
                
                # === Build cluster summary ===
                cluster_summary = labeled_df.groupby("Cluster").agg(
                    N_Trajectories=("Correct", "count"),
                    N_Correct=("Correct", "sum")
                ).reset_index()
                
                
                cluster_summary["GlobalPctCorrect"] = cluster_summary["N_Correct"] / total_trials_global
                

                cluster_summary["PctCorrect"] = cluster_summary["N_Correct"] / cluster_summary["N_Trajectories"]


                for entry in results:
                    if entry["Session"] == session_date:
                        cluster = entry["Cluster"]
                        match = cluster_summary[cluster_summary["Cluster"] == cluster]
                        if not match.empty:
                            entry["PctCorrect"] = match["PctCorrect"].values[0]
                            entry["GlobalPctCorrect"] = match["GlobalPctCorrect"].values[0]
                            entry["N_Correct"] = match["N_Correct"].values[0]
                        else:
                            entry["PctCorrect"] = np.nan
                            entry["GlobalPctCorrect"] = np.nan
                            entry["N_Correct"] = np.nan
                

            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

    print("Not enough trajectories to sample from.")
results_df = pd.DataFrame(results)
results_df = results_df[results_df["Spatiotemporal Variability"] <= 2000]


import seaborn as sns
import matplotlib.pyplot as plt


### manual renaming of identified clusters per mouse 
rename_map = {
    "Cluster 0": "Low Waterport",
    "Cluster 1": "unclassfiable - mostly high water port",
    "Cluster 2": "High Waterport",
    "Cluster 3": "Exploratory"
}

results_df.to_csv("/Volumes/Expansion/sem-data/untitled folder/M1791/variability_cluster_correctness-1797.csv", index=False)

results_df["Cluster Label"] = results_df["Cluster"].map(rename_map)

results_df_sorted = results_df.sort_values(by="Session")

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=results_df_sorted,
    x="Session",
    y="Spatiotemporal Variability",
    hue="Cluster Label",
    marker="o",
    palette= {
    "Low Waterport": "green",
    "unclassfiable - mostly high water port": "red", # change this, depending on the colors detected
    "High Waterport": "blue",
    "ow Waterport" : "blue"
} 
)

plt.xticks(rotation=45)
plt.title("Spatiotemporal Variability per Cluster Over Time")
plt.ylabel("Spatiotemporal Variability")
plt.xlabel("Session Date")
plt.tight_layout()
plt.legend(title="Cluster")
plt.show()


