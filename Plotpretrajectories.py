#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-Trial (3000 ms) Trajectory Script with Background Image
Scans 'top' folders, finds behavior CSVs & DLC coordinate CSVs,
and plots 3s of nose trajectories before each trial, using an
image as the background.

Assumes:
  1) BehaviorCSVReader + CSVDataReader 
  2) Body part named "Nose" 
  3) recorded at 25 FPS 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  
from BehaviorCSVReader import BehaviorCSVReader
from CSVDataReader import CSVDataReader

class PretrialTrajectoryAnalyzer:
    def __init__(self, base_path, top_img_path=None, dlc_fps=25):
        self.base_path = base_path
        self.top_img_path = top_img_path
        self.dlc_fps = dlc_fps
    def extract_pretrial_trajectories_3s(self, dlc_array, trial_times):
        """
        For each (trial_timestamp_ms, side) in trial_times, gather frames in 
        the interval [start - 3000, t_start). Return a list of dicts:

        """
        dt_ms = 1000.0 / self.dlc_fps
        n_frames = dlc_array.shape[0]
        frame_times = np.arange(n_frames) * dt_ms
    
        output = []
        for (t_start, side) in trial_times:
            t_min = t_start - 3000 # set the desired time interval
            mask = (frame_times >= t_min) & (frame_times < t_start)
            subset_coords = dlc_array[mask]
            subset_times = frame_times[mask]
    
            if subset_coords.shape[0] > 0:
                output.append({
                    'TrialStart': t_start,
                    'TrialSide':  side,
                    'Frames':     subset_coords,
                    'FrameTimes': subset_times
                })
    
        return output
    
    
    def plot_pretrial_trajectories_3s(self, pretrial_data, file_name, mouse_name):
        """
        Plots the 3s pre-trial trajectories for each trial with a background image.
    
        :param pretrial_data: list of dicts from extract_pretrial_trajectories_3s()
        :param top_img_path:  path to a top-view image to display as background
        """
        if not pretrial_data:
            print("No pre-trial data to plot.")
            return
    

        fig, ax = plt.subplots(figsize=(10, 6))
    
        #background image
        if self.top_img_path and os.path.isfile(self.top_img_path):
            topview = Image.open(self.top_img_path)
            # For example, match the same dimension approach as SkilledMovementSwitchingTask
            width, height = 1200, 600
            topview = topview.resize((width, height), Image.Resampling.LANCZOS)
    
            ax.imshow(topview)         
            ax.set_xlim([0, width])    # Match coordinate system
            ax.set_ylim([0, height])

        else:
            print("No valid background image path provided or file not found.")
    
        # Colors for side=0 or side=1
        side_colors = {0: 'blue', 1: 'red'}
    
    
        #used_labels = {'Start': False, 'End': False} --- if start and end-points are desired 
    
        for trial_dict in pretrial_data:
            frames = trial_dict['Frames']
            side   = trial_dict['TrialSide']
            color  = side_colors.get(side, 'gray')
    
            x = frames[:, 0]
            y = frames[:, 1]
    
            # Plot the trajectory line
            label = "Low Waterport" if side == 0 else "High Waterport"
            ax.plot(x, y, color=color, alpha=0.5, label=label)
    
            # Mark start point (the first coordinate ~ -3000 ms)
            #start_x, start_y = x[0], y[0]
            #if not used_labels['Start']:
             #   ax.scatter(start_x, start_y, color='black', s=20, zorder=3, label="Start")
            #    used_labels['Start'] = True
            #else:
            #    ax.scatter(start_x, start_y, color='black', s=20, zorder=3)
    
            # Mark end point (the last coordinate ~ 0 ms)
            #end_x, end_y = x[-1], y[-1]
           # if not used_labels['End']:
             #   ax.scatter(end_x, end_y, color='green', s=20, zorder=3, label="End")
             #   used_labels['End'] = True
            #else:
             #   ax.scatter(end_x, end_y, color='green', s=20, zorder=3)
    
        # Remove repeated legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique_pairs = dict(zip(labels, handles))
        ax.legend(unique_pairs.values(), unique_pairs.keys(), loc='best')
    
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Trajectories of Mouse 00{mouse_name} - {file_name} 3000ms before trial")
    
        plt.tight_layout()
        plt.show()
    
    
    def find_top_folders(self):
        top_folders = []
        for root, dirs, _ in os.walk(self.base_path):
            if "top" in dirs:
                top_folders.append(os.path.join(root, "top"))
        return top_folders
    
    
    def process_all_sessions(self):
        """
        Automatically finds top-folders,
        looks for a DLC file (ends with '_filtered.csv'),
        and a single behavior CSV in the parent folder.
        """
        top_folders = self.find_top_folders()

    
        for folder in top_folders:
            print(f"\nProcessing folder: {folder}")
    
            # Identify all DLC files that end with 'filtered.csv'
            for file in os.listdir(folder):
                if file.startswith("._") or not file.endswith("filtered.csv"):
                    continue
    
                coord_csv = os.path.join(folder, file)
                file_name = file[6:12]
                mouse_name = file[4]# e.g., "MO406-" etc. --- for the correct naming of the plots
    
                sensor_folder = os.path.dirname(folder)
                sensor_csv = next(
                    (f for f in os.listdir(sensor_folder) if f.endswith(".csv") and not f.startswith("._")),
                    None
                )
                if sensor_csv is None:
                    print("No sensor CSV found in", sensor_folder)
                    continue
    
                #sensor_csv_path = os.path.join(sensor_folder, sensor_csv)
    
                # 1) Behavior data
                br = BehaviorCSVReader(sensor_folder, sensor_csv)
                behavior_df, info_sensors = br.read_behavior_csv()
                trial_times = info_sensors.get('TrialTimes', [])
                if not trial_times:
                    print("No trial timestamps found in the behavior CSV.")
                    continue
    
                # 2) DLC data
                cdr = CSVDataReader(coord_csv)
                body_parts = cdr.get_data_from_csv(exclude_list=['bodyparts'], omit_prediction=False)
                if "Nose" not in body_parts:
                    print(f"Nose data not found in {coord_csv}. Keys: {list(body_parts.keys())}")
                    continue
    
                dlc_array = body_parts["Nose"]  # Nx2 or Nx3
    
                # 3) Extract 3000 ms pre-trial
                pretrial_data_3s = self.extract_pretrial_trajectories_3s(dlc_array, trial_times)
                
                # 4) Plot, optionally including background
                print(f"Plotting pre-trial data for file: {file_name} (3s before each trial)")
                self.plot_pretrial_trajectories_3s(pretrial_data_3s, file_name, mouse_name)


