"""
The Wireless Intelligence Lab

DeepSense 6G
http://www.deepsense6g.net/

Description: Dataset loader function for radar data
Author: Umut Demirhan
Date: 10/29/2021
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat


def load_radar_data(root_dir, csv_file, radar_column='unit1_radar_1', label_column='beam_index_1'):
    csv_file = pd.read_csv(os.path.join(root_dir, csv_file))
    
    X = loadmat(os.path.abspath(root_dir +  csv_file[radar_column][0]))['data'] # Load the first data point
    X = np.zeros((len(csv_file),) + X.shape, dtype=X.dtype)
    for i in tqdm(range(len(csv_file))):
        X[i] = loadmat(os.path.abspath(root_dir +  csv_file[radar_column][i]))['data']
    y = np.array(csv_file[label_column])
    return X, y