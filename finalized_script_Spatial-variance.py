#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 17:29:19 2025

@author: felix
"""
import matplotlib.pyplot as plt 
import numpy as np
import os
from AnalyzeExploration import AnalyzeExploration
from collections import defaultdict
 
class ExplorationAnalyzerTemporal:
    def __init__(self, width2, height2, grid_size=10):
        self.width2 = width2
        self.height2 = height2
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)

    def _map_to_coarse_grid(self, x, y):
        col_bin = int((x / self.width2) * self.grid_size)
        row_bin = int((y / self.height2) * self.grid_size)
        col_bin = min(max(col_bin, 0), self.grid_size - 1)
        row_bin = min(max(row_bin, 0), self.grid_size - 1)
        return row_bin, col_bin

    def update_grid(self, x_coords, y_coords):
        for x, y in zip(x_coords, y_coords):
            row_bin, col_bin = self._map_to_coarse_grid(x, y)
            self.grid[row_bin, col_bin] += 1

    def compute_spatial_distribution_variance(self):
        total_visits = np.sum(self.grid)
        if total_visits == 0:
            print("No visits recorded, variance undefined.")
            return None

        normalized_distribution = self.grid.flatten() / total_visits
        variance = np.var(normalized_distribution)
        return variance
    
    '''def plot_visit_distribution(self):
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(self.grid.flatten())), self.grid.flatten())
        plt.title("Visit Distribution Across Grid Cells")
        plt.xlabel("Grid Cells")
        plt.ylabel("Number of Visits")
        plt.tight_layout()
        plt.grid(False)
        plt.show()
        plt.close()'''
        
def find_top_folders(base_path):
    top_folders = []
    for root, dirs, _ in os.walk(base_path):
        if "top" in dirs:
            top_folders.append(os.path.join(root, "top"))
    return top_folders


def process_and_plot_variance(base_path):
    variances = []
    file_names = []

    top_folders = find_top_folders(base_path)

    for folder in top_folders:
        print(f"Processing folder: {folder}")

        for file in os.listdir(folder):
            if file.startswith("._") or not file.endswith("filtered.csv"):
                continue

            csv_path = os.path.join(folder, file)
            print(f"Processing CSV: {csv_path}")

            exploration = AnalyzeExploration(csv_path, '/Volumes/Expansion/New_cleaned_data/box.jpg')
            binary_grid = exploration.makegrids(csv_path)

            if binary_grid is not None:
                coords = np.argwhere(binary_grid.to_numpy() > 0)
                if coords.size == 0:
                    print(f"No visits recorded in {csv_path}")
                    continue

                width, height = binary_grid.shape[::-1]
                temporal_analyzer = ExplorationAnalyzerTemporal(width, height, grid_size=10)
                temporal_analyzer.update_grid(coords[:, 1], coords[:, 0])
                session_variance = temporal_analyzer.compute_spatial_distribution_variance()

                if session_variance is not None:
                    variances.append(session_variance)
                    file_names.append(file[6:12])  # file name
                    print(f"Variance for {file[:12]}: {session_variance}")

    if not variances:
        print("No variance data collected.")
        return


    sorted_indices = np.argsort(file_names)
    file_names = [file_names[i] for i in sorted_indices]
    variances = [variances[i] for i in sorted_indices]

    # Stats
    variances_array = np.array(variances)
    mean_variance = np.mean(variances_array)
    std_variance = np.std(variances_array)



    # Step 1: Aggregate variances by date
    date_variance_dict = defaultdict(list)
    
    for file_name, var in zip(file_names, variances):
        # Assuming file_name = "M1791_250423"
        session_date = file_name.split('_')[-1]  # extract '250423'
        date_variance_dict[session_date].append(var)
    
    # Step 2: Compute means and std deviations
    session_dates = sorted(date_variance_dict.keys())
    mean_variances = [np.mean(date_variance_dict[date]) for date in session_dates]
    std_variances = [np.std(date_variance_dict[date]) for date in session_dates]
    
    # Step 3: Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(session_dates))
    plt.plot(x, mean_variances, linestyle='-', linewidth=2, color='black', label='Mean Spatial Variance')
    plt.fill_between(x,
                     np.array(mean_variances) - np.array(std_variances),
                     np.array(mean_variances) + np.array(std_variances),
                     color='gray', alpha=0.3)
    plt.xticks(x, session_dates, rotation=45, ha="right", fontsize=10)
    plt.axhline(np.mean(mean_variances), color='black', linestyle='--', linewidth=1.5, label='Overall Mean')
    
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Session Date", fontsize=12)
    plt.ylabel("Spatial Variance", fontsize=12)
    plt.ylim(0, 0.00045)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.grid(False)
    plt.show()

process_and_plot_variance(base_path='/Volumes/Expansion/Usable_data/M1791/')
process_and_plot_variance(base_path='/Volumes/Expansion/Usable_data/M1792/')
process_and_plot_variance(base_path='/Volumes/Expansion/Usable_data/M1794/')
process_and_plot_variance(base_path='/Volumes/Expansion/Usable_data/M1795/')
process_and_plot_variance(base_path='/Volumes/Expansion/Usable_data/M1796/')
process_and_plot_variance(base_path='/Volumes/Expansion/Usable_data/M1797/') # needs multiple sessions per date to compute std-differences, so multiple mice per group