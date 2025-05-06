# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:03:00 2024

@author: Gulraiz Iqbal Choudhary
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
            for  line_number, line in enumerate(file, 1):
                if 'Trial' in line:
                    parts = line.split(',')
                    time_stamp = int(parts[0])
                    side = int(parts[1].split(':')[1].strip())
                    self.trial_times.append((time_stamp, side))
                elif 'slow' in line:
                    continue  # Handle 'slow' entries if needed
                elif 'CAP' in line:
                    self.thres = int(line.split('=')[1].strip())
                elif 'SENSOR3_HOLD' in line:
                    self.holding_time = int(line.split('=')[1].strip())
                elif 'STARTING_MODE' in line:
                    self.training_type = int(line.split('=')[1].strip())
                elif 'Starting protocol'in line:
                    self.protocol_index = line_number+1
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
                         skiprows=self.protocol_index, skipfooter=4, engine='python')#warn_bad_lines=True)  # Warn about skipped lines)
        data = []
        current_trial = None
        for index, row in df.iterrows():
            if pd.isna(row[1]) is False and re.match(r'\s*Trial:\s*\d+', row[1]):
                current_trial = int(row[1].split(':')[1].strip())
                continue
            new_row = list(row) + [current_trial]
            current_trial = None
            data.append(new_row)

        new_columns = ['Timestamp', 'Right Sensor', 'Left Sensor', 'Middle Sensor', 'Status', 'Trial']
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
            'ProcessedData': self.processed_df
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


