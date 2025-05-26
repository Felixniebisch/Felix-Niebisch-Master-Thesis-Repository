# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:03:00 2024

@author: Gulraiz Iqbal Choudhary

has been updated by me ( Felix Niebisch ) to also include the new assigned columns

"""

import pandas as pd
import re
import os
import numpy as np

class BehaviorCSVReader:
    def __init__(self, data_dir, csv_file):
        self.data_dir = data_dir
        self.csv_file = csv_file
        
        # Initialize class members
        self.trial_times = []
        self.timestamps = []
        self.first_touch = []
        self.end_times = []
        self.latency_corr = []
        self.latency_incorr = []
        self.correct_trials = []
        self.incorrect_trials = []
        self.change_choice = []
        self.trial_windows = []
        self.left_trial_windows = []
        self.right_trial_windows = []
        self.s1 = []
        self.s2 = []
        self.s3 = []
        self.holding_time = None
        self.thres = None
        self.training_type = None
        self.processed_df = None
        self.info_sensors = None
        
        self.protocol_index = None
        
    def read_behavior_csv(self):
        # Read the CSV file line by line
        with open(os.path.join(self.data_dir, self.csv_file), 'r') as file:
            lines = list(file)
            for i, line in enumerate(lines):
                if 'Trial' in line:
                    parts = line.split(',')
                    time_stamp = int(parts[0])
                    side = int(parts[1].split(':')[1].strip().replace('"', ''))
                    self.trial_times.append((time_stamp, side))
                    
                    # Scan forward to find first sensor2 touch
                    first_touch_time = None
                    first_sensor = None
                    for future_line in lines[i+1:]:             
                        future_parts = future_line.strip().split(',')
                        try:
                            event_time = int(future_parts[0])
                            l2 = int(future_parts[4])  # assuming column 5 is Left_sensor
                            r2 = int(future_parts[5])  # assuming column 6 is Right_sensor
                        
                        except (ValueError, IndexError):
                            continue 
        
                        if l2 == 1:
                            first_touch_time = event_time
                            
                            
                            first_sensor = 0
                            
                            if first_sensor == side:
                                self.correct_trials.append(1)
                                self.incorrect_trials.append(0)
                                latency = first_touch_time - time_stamp
                                self.latency_corr.append(latency) # latency to reach the left waterport
                                self.end_times.append(first_touch_time)
                                self.trial_windows.append((time_stamp, first_touch_time, side))
                                self.left_trial_windows.append((time_stamp, first_touch_time, side))
                            else:
                                self.incorrect_trials.append(1)
                                self.correct_trials.append(0)
                                latency = first_touch_time - time_stamp
                                self.latency_incorr.append(latency)
                                self.end_times.append(first_touch_time)
                                self.trial_windows.append((time_stamp, first_touch_time, side))
                                self.left_trial_windows.append((time_stamp, first_touch_time, side))
                            break
                        elif r2 == 1:
                            first_touch_time = event_time
                            
                            first_sensor = 1
                            
                            if first_sensor == side:
                                self.correct_trials.append(1)
                                self.incorrect_trials.append(0)
                                latency = first_touch_time - time_stamp
                                self.latency_corr.append(latency) 
                                self.end_times.append(first_touch_time)
                                self.trial_windows.append((time_stamp, first_touch_time, side))
                                self.right_trial_windows.append((time_stamp, first_touch_time, side))
                            else:
                                self.incorrect_trials.append(1)
                                self.correct_trials.append(0)
                                latency = first_touch_time - time_stamp
                                self.latency_incorr.append(latency) 
                                self.end_times.append(first_touch_time)
                                self.trial_windows.append((time_stamp, first_touch_time, side))
                                self.right_trial_windows.append((time_stamp, first_touch_time, side))
                            break
                elif 'slow' in line:
                    continue  # Handle 'slow' entries if needed
                elif 'CAP' in line:
                    self.thres = int(line.split('=')[1].strip())
                elif 'SENSOR3_HOLD' in line:
                    self.holding_time = int(line.split('=')[1].strip())
                elif 'STARTING_MODE' in line:
                    self.training_type = int(line.split('=')[1].strip())
                elif 'Starting protocol'in line:
                    self.protocol_index = i + 1
                else:
                    parts = line.split(',')
                    if len(parts) == 5:
                        self.timestamps.append(int(parts[0].strip()))
                        self.s1.append(int(parts[1].strip()))
                        self.s2.append(int(parts[2].strip()))
                        self.s3.append(int(parts[3].strip()))

        # Create a DataFrame for sensor data
        sensor_data = pd.DataFrame({
            'Timestamp': self.timestamps,
            'Right Sensor': self.s1,
            'Left Sensor': self.s2,
            'Middle Sensor': self.s3
        })
        sensor_data.set_index('Timestamp', inplace=True)

        # Convert to numpy arrays if needed for further processing
        self.s1 = np.array(self.s1)
        self.s2 = np.array(self.s2)
        self.s3 = np.array(self.s3)

        # Generate the processed DataFrame as done in the earlier code
        df = pd.read_csv(os.path.join(self.data_dir, self.csv_file), header=None, 
                         skiprows=self.protocol_index, skipfooter=4, engine='python', on_bad_lines='skip')#warn_bad_lines=True)  # Warn about skipped lines)
        data = []
        current_trial = None
        
        for index, row in df.iterrows():
            # Clean second column if it exists
            if len(row) > 1 and pd.notna(row.iloc[1]):
                value = str(row.iloc[1]).replace('"', '').strip()
                if re.match(r'^Trial:\s*\d+', value):
                    try:
                        trial_num = int(re.search(r'Trial:\s*(\d+)', value).group(1))
                        current_trial = trial_num
                    except Exception as e:
                        print(f"Failed to extract trial number at row {index}: {value}")
                    continue  # Skip this row — it's not data
        
            # Sensor data row
            row_values = list(row)
            if len(row_values) < 7:
                print(f"Skipping short row at index {index}: {row_values}")
                continue
        
            new_row = row_values[:7] + [current_trial]
            current_trial = None  # Only apply to next row
            data.append(new_row)

        new_columns = ['Timestamp', 'Right Sensor', 'Left Sensor', 'Middle Sensor','Leftsensor2', 'Rightsensor2', 'Status', 'Trial'] # this needs to be adjusted in correspondence to the real new sensor
        self.processed_df = pd.DataFrame(data, columns=new_columns)
        self.processed_df.set_index('Timestamp', inplace=True)
        self.processed_df['Status'] = self.processed_df['Status'].astype(str).fillna('')

        # Process the 'Status' column
        def process_status(status):
            status = status.strip()
            if len(status) < 6:
                return False, False, False, False, False, None
            right_camera = status[0] == 'T'
            left_camera = status[1] == 'T'
            top_camera = status[2] == 'T'
            auditory_cue = status[3] == 'T'
            grey_period = status[4] == 'G'
            go_cue = status[5] if status[5].isdigit() else None
            return right_camera, left_camera, top_camera, auditory_cue, grey_period, go_cue

        status_columns = ['Right Camera', 'Left Camera', 'Top Camera', 'Auditory Cue', 'Grey Period', 'Go Cue']
        status_data = self.processed_df['Status'].apply(process_status)
        status_df = pd.DataFrame(status_data.tolist(), index=self.processed_df.index, columns=status_columns)
        self.processed_df = pd.concat([self.processed_df, status_df], axis=1)

        # Clean sensor columns and convert 'Timestamp' index to numeric values
        def clean_numeric_column(col):
            col = col.replace(r'[^\d.-]', '', regex=True)
            return pd.to_numeric(col, errors='coerce')

        sensor_columns = ['Right Sensor', 'Left Sensor', 'Middle Sensor']
        self.processed_df[sensor_columns] = self.processed_df[sensor_columns].apply(clean_numeric_column)
        self.processed_df.index = pd.to_numeric(self.processed_df.index, errors='coerce').astype('Int64')
        self.processed_df = self.processed_df[~self.processed_df.index.isna()]

        # Save InfoSensors as a dictionary and store in a pickle file
        self.info_sensors = {
            'TrialTimes': self.trial_times,
            'Timestamps': self.timestamps,
            'RightSensor': self.s1,
            'LeftSensor': self.s2,
            'MiddleSensor': self.s3,
            'HoldingTime': self.holding_time,
            'SensorThreshold': self.thres,
            'TrainingType': self.training_type,
            'ProcessedData': self.processed_df,
            'CorrectTrials': self.correct_trials,
            'IncorrectTrials': self.incorrect_trials,
            'LatencyCorrect': self.latency_corr,
            'LatencyIncorrect': self.latency_incorr,
            'EndTimes': self.end_times,
            'TrialWindows': self.trial_windows,
            'LeftWindows' : self.left_trial_windows,
            'RightWindows': self.right_trial_windows
        }
        
        save_path = os.path.join(self.data_dir, 'InfoSensors.pkl')
        #self.processed_df = pd.DataFrame(self.processed_df)
        #self.processed_df.head()
        #new_df = self.processed_df
        '''Trial_number = 0
        for i in range(1, 10):
            Trial_number += 1
            new_df.to_csv(f'/Volumes/Expansion/infosensors_df/Trial1-mouse 009/test{Trial_number}.csv')'''
        #print(self.info_sensors)
        #self.info_sensors = pd.DataFrame(self.info_sensors)
        #self.info_sensors.head()
        #sensor_df = self.info_sensors
        #sensor_df.to_csv('/Volumes/Expansion/infosensors_df/Trial1-mouse 009/sensors1.csv')

        pd.to_pickle(self.info_sensors, save_path)
  
        return self.processed_df, self.info_sensors

