#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:20:54 2025

@author: felix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os        
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

class HoldingTimeAnalysis:
    def __init__(self, threshold=30):
        self.threshold = threshold
        self.grouped_intervals = None

    def load_and_prepare_data(self, csv_path):
        """
        Loads and prepares data from a CSV file, skipping unnecessary text parts.
        """
        try:
            print(f' we are now processing: {csv_path}')
            # Read the CSV file
            df = pd.read_csv(
                csv_path, 
                skiprows=71,  
                engine='python',
                on_bad_lines='skip',
                header=None  
            )
            
           
            #print("Detected Columns Before Renaming:", list(df.columns))

            # Keep only the first 5 columns 
            if df.shape[1] > 5:
                df = df.iloc[:, :5]
                
            # Rename columns to the expected names
            df.columns = ['Timestamp', 'Right sensor', 'Left sensor', 'Middle_sensor', 'status'] #change the random names to the respective variables later on 

            # Convert the 'Timestamp' column to numeric
            df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
            df.dropna(subset=['Timestamp'], inplace=True)  # Remove rows where Timestamp couldn't be converted
            df['Timestamp_ms'] = df['Timestamp'].astype(int)

            # Print detected columns after renaming for debugging
            #print("Detected Columns After Renaming:", list(df.columns))

            df = df[~df.apply(lambda row: row.astype(str).str.contains(r'Trial:\s*[01]', regex=True)).any(axis=1)]
            df = df[~df.apply(lambda row: row.astype(str).str.contains(r'loop too slow', regex=True)).any(axis=1)]
            return df

        except Exception as e:
            print(f"Failed to load {csv_path}: {e}")
            return None
        
    def detect_intervals(self, df, file_name):
       '''This function first creates a new Dataframe corresponding to the Timestamp and the middle sensor values and subsequently reads out 
       all the intervals where the threshold value on the middle sensor has been reached. Then all intervals that are longer than 100 ms 
       are added into a dictionary with their corresponding starting time, ending time and total interval length'''
       
       #skip_status = {'	xxxxGx', 'xxTxGx', 'xxT1Gx', 'xxx2Gx', 'xxx1Gx' } # manually skipping rows
       Trial_threshold = 50 # the threshold value that is needed to initiate a trial
       Middle = df['Middle_sensor']
       Timestamp = df['Timestamp_ms']
       Status = df['status']
       df = pd.concat([Middle, Timestamp, Status], axis=1)
       df = pd.DataFrame(df)
       starting_time = None
       processed_dict = []
       first_row_below = False
       count = 0
       for index, row in df.iterrows():
            status = row['status']
            if pd.isna(status):
                continue
            else:
               status = row['status'].strip()

           
            if row["Middle_sensor"] >= self.threshold:
                if starting_time is None:
                    starting_time = row["Timestamp_ms"]
                    first_row_below = False
                continue
            
            if starting_time is not None:
        
                if not first_row_below:
                    first_row_below = True
                    continue
        
                # Second below-threshold row - check status
                if status[4] == 'G':
                    # Cancel interval due to invalid status
                    continue
        
                # Otherwise, end interval normally
                ending_time = row["Timestamp_ms"]
                interval = ending_time - starting_time
        
                if interval >= Trial_threshold:
                    
                    print(f' INTERVAL DETECTED {count}\n\n')
                    processed_dict.append({
                        "Start_time": starting_time,
                        "End_time": ending_time,
                        "Interval": interval
                    })
                    count += 1
                
                    print(f'The starting time for {file_name} interval {count} is {starting_time}\n and has the ending time: {ending_time}\n')
        
                # Reset tracking
                starting_time = None
                first_row_below = False
       print('The file {file_name} has been completed\n\n')
       

       self.processed_df = pd.DataFrame(processed_dict)
       
       
       if not self.processed_df.empty:
           self.processed_df['Bin_20ms'] = (self.processed_df['Interval'] // 20) * 20
       
           # Count how many intervals fall into each bin
           bin_counts = self.processed_df['Bin_20ms'].value_counts().sort_index()
       
           # Just print the aggregated version
           print(f'The file {file_name} has {len(processed_dict)} trials longer than 500 ms.') # adjust <ms> if needed
       
           # Plot directly from the bin_counts
           axis = bin_counts.plot(
               kind="bar", figsize=(10, 5), legend=False
           )
           axis.set_xlabel("Holding time duration displayed in 20 ms bins")
           axis.set_ylabel("The number of events these holding times have occurred")
           axis.set_title(file_name)
           axis.set_ylim(0, 200)  
           plt.tight_layout()
           plt.show()  




    def plot_Intervals(self, file_name):
        if not self.processed_df.empty:
            median = self.processed_df["Interval"].median()
    
            # Plot the boxplot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.boxplot(self.processed_df["Interval"], vert=True, patch_artist=True)
            ax.set_ylabel("Duration (ms)")
            ax.set_title(f"Interval Duration Distribution – {file_name}")
            ax.set_xticks([1])
            ax.set_xticklabels(["Intervals"])
    
            # Show median in the legend 
            median_text = f"Median: {median:.1f} ms"
            ax.legend([median_text], loc='upper right', frameon=False)
    
            plt.tight_layout()
            plt.show()
            

            




    def process_all_subfolders(self, base_path):
        all_intervals = []
        box_intervals = []

        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)

            if os.path.isdir(folder_path):
                print(f"\n entering folder: {folder_path}")

                for file in os.listdir(folder_path):
                    if file.endswith(".csv") and not file.startswith("._"):
                        sensor_csv_path = os.path.join(folder_path, file)
                        print(f" Processing CSV: {sensor_csv_path}")

                        df = self.load_and_prepare_data(sensor_csv_path)
                        file_name = file[13:19]  # Adjust this slicing as needed

                        if df is not None:
                            self.detect_intervals(df, file_name)
                            self.plot_Intervals(file_name)

                            median_value = self.processed_df["Bin_20ms"].median()
                            medians = {"Interval": median_value, "File": file_name}
                            all_intervals.append(medians)
                            box_intervals.append(self.processed_df["Interval"])
    
    
    
        if box_intervals:
            fig, axi = plt.subplots(figsize=(8, 6))
            axi.boxplot(box_intervals, vert=True, patch_artist=True)
            axi.set_xlabel("Files")
            axi.set_ylabel("Holding Time Duration (ms)")
            axi.set_title("Boxplot of Holding Time for All Files")
            axi.legend(f" Median = {median_value}")
            plt.tight_layout()
            plt.show()
                
        
        if box_intervals:
            # Create a DataFrame where each column is a trial (i.e. a file)
            intervals_df = pd.DataFrame(box_intervals).transpose()
        
            # Now compute real Q25, median, Q75 across rows (i.e. trial-wise)
            median = intervals_df.median(axis=0)
            q25 = intervals_df.quantile(0.25, axis=0)
            q75 = intervals_df.quantile(0.75, axis=0)
        
            # Trial index
            trials = np.arange(1, len(median) + 1)
        
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(trials, median, color='black', linewidth=2, label='Median Holding Duration')
            ax.fill_between(trials, q25, q75, color='gray', alpha=0.5, label='25–75% IQR')
        
            # Labels and style
            ax.set_xlabel("Trial", fontsize=11)
            ax.set_ylabel("Holding Duration (ms)", fontsize=11)
            ax.set_ylim(0, 1000)
            ax.set_title("Per-Trial Median Holding Duration with IQR", fontsize=12)
        
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction='out', length=4, width=1)
            ax.grid(True, linestyle='--', alpha=0.5)
        
            plt.legend()
            plt.tight_layout()
            plt.show()
    
            # Create a DataFrame where each column is a trial (i.e. a file)
            intervals_df = pd.DataFrame(box_intervals).transpose()
        
            # Compute real mean and standard deviation across rows (per trial block)
            mean = intervals_df.mean(axis=0)
            std = intervals_df.std(axis=0)
        
            # Trial index (same length as number of trials/files)
            trials = np.arange(1, len(mean) + 1)
        
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(trials, mean, color='black', linewidth=2, label='Mean Holding Duration')
            ax.fill_between(trials, mean - std, mean + std, color='lightgray', alpha=0.7, label='±1 SD')
        
            # Labels and style
            ax.set_xlabel("Trial", fontsize=11)
            ax.set_ylabel("Holding Duration (ms)", fontsize=11)
            ax.set_title("Per-Trial Mean Holding Duration with ±1 SD", fontsize=12)
        
            ax.set_ylim(0, 1000)  # ✅ force x-axis from 0 to 1000
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction='out', length=4, width=1)
            ax.grid(True, linestyle='--', alpha=0.5)
        
            plt.legend()
            plt.tight_layout()
            plt.show()        
            
        if all_intervals:
            
            all_medians_df = pd.DataFrame(all_intervals)
    
            split_point = len(all_medians_df) // 2
            all_medians_df["Trial"] = all_medians_df.index + 1
            all_medians_df["Phase"] = ["Early" if i < split_point else "Late" for i in all_medians_df.index]
            
            # Step 2: Use a rolling window to compute central tendency and spread
            # (if data is too noisy, you can skip rolling and group instead)
            window = 5  # You can adjust this
            rolling = all_medians_df["Interval"].rolling(window=window, center=True)
            
            median = rolling.median()
            q25 = rolling.quantile(0.25)
            q75 = rolling.quantile(0.75)
            
            # Step 3: Plot
            plt.figure(figsize=(10, 5))
            plt.plot(all_medians_df["Trial"], median, color='blue', label='Median Holding Duration')
            plt.fill_between(all_medians_df["Trial"], q25, q75, color='blue', alpha=0.2, label='25–75% IQR')
            
            # Optional: Add vertical line to indicate phase transition
            #plt.axvline(split_point + 0.5, linestyle='--', color='black', label='Early/Late Split')
            
            # Labels, legend, etc.
            plt.xlabel("Trial Block Index")
            plt.ylabel("Holding Duration (ms)")
            plt.title("Median Holding Duration per Trial with IQR")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.show()
        





    






    





#csv = pd.read_csv('/Volumes/Expansion/tests/csv_new/csv2.csv')




#Threshold = 30 





#proc_df.plot(x= "Start_time", y="Interval", kind="bar")
#plt.show

#print(proc_df)
#array = np.array(Intervals)
#print(array.argmax)
long_intervals = []
#print(type(Intervals))

#for interval in array:

#    if interval >= 200:
#        long_intervals.append(interval)
    
    
#print(f' The long interval is : {long_intervals}')

#mean_in = sum(long_intervals) /len(long_intervals)
#print(len(long_intervals))
#print(mean_in)           
        
       # print(lenrow['Timestamp_ms'])
    
    #print(f'{row["Timestamp_ms"]}: Middle value: {row["Middle_sensor"]}')
        

