# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:42:42 2024

@author: Gulraiz.Iqbal
"""

import pandas as pd
import numpy as np
import sys
import os

class CSVDataReader:
    def __init__(self, file_path):
        """
        Initializes the CSVDataReader with the file path.
        """
        self.file_path = file_path
        self.data = None

    def get_header_name_start_index(self, column_headers, header_name):
        """
        Gets the first index of a header name from a list of column headers.
        Ex:
            column headers = bodyparts,Nose,Nose,Nose,LeftEar,LeftEar,LeftEar,RightEar,RightEar,RightEar
        header name = Nose
        should return 1

        :param column_headers: a string with the comma seperated column headers
        :param header_name: the column header looked for
        :return: The first index the header_name is found in
        """
        for index, header in enumerate(column_headers):
            if header == header_name:
                return index

        print("Did not find a column header named: " + header_name + " inside of headers: ")
        print(column_headers)
        sys.exit(1)


    def get_body_parts_from_csv_file(self, csv_file, body_parts, omit_prediction):
        """
        Gets body parts data from a csv file.
        :param csv_file: the file to look for the body parts in
        :param body_part: the body part that we want data from
        :param omit_predicition: if the prediction value should be omitted or not
        :return: a matrix with the body parts x,y, and prediction value if it is not omitted
        """
        rows_in_csv = csv_file.shape[0]
        # The whole indexing here is kind of confusing, but checked that it takes the first and the last values. So should be correct.
        nr_of_data_rows = rows_in_csv - 2  # total numbers minus the number of headers
        data_start_index = 2  # should start from index 2

        header = list(csv_file.iloc[0])
        body_part_start_index =self.get_header_name_start_index(header, body_parts)
        nr_of_elements_to_capture = 3

        if omit_prediction:
            nr_of_elements_to_capture = nr_of_elements_to_capture - 1

        data_matrix = np.zeros((nr_of_data_rows, nr_of_elements_to_capture), dtype=np.float64)
        data_matrix = csv_file.iloc[data_start_index:, body_part_start_index:body_part_start_index +
                                    nr_of_elements_to_capture].to_numpy().astype(np.float64)

        return data_matrix


    def get_data_from_csv(self, exclude_list, omit_prediction):
        """
        Gets all the body part data from the file specified.

        :param file_path: path to the file that should be read
        :param exclude_list: headers to exclude
        :param omit_prediction: if the prediction value should be omitted or not
        :return: a dictionary with the body part name as key, and a array of the data we want
        """
        csv_file = pd.read_csv(self.file_path, low_memory=False)

        body_parts_data = dict()

        all_headers = set(list(csv_file.iloc[0]))
        searched_headers = list(all_headers.difference(exclude_list))

        for body_part in searched_headers:
            body_parts_data[body_part] = self.get_body_parts_from_csv_file(csv_file, body_part, omit_prediction)

        return body_parts_data


#go one step backward
os.chdir('..')
current_path = '/Volumes/Expansion/Felix'

mouse_name = "Mouse2024006"

session_name = "Mouse2024006-240624-100847"

# # uncomment this code to analyze a single Mouse kinematics
#path_file = '/Volumes/Expansion/Felix/Mouse2024006/Mouse2024006-240624-100847/top/MO406-240624-0936-topDLC_resnet50_ObjectInteractionOct12shuffle1_200000_filtered.csv'

# # Initialize the reader with a file path
#csv_reader = CSVDataReader(path_file)

#should_omit_prediction_values = True
#exclude_list = ['bodyparts']
 

# # Read the CSV file
#data = csv_reader.get_data_from_csv(exclude_list, should_omit_prediction_values)

#print(data['Nose'][:,1])
#data.to_csv(/Volumes/Expansion/random_dfs)
