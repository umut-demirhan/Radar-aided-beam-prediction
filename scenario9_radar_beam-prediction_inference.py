"""
The Wireless Intelligence Lab

DeepSense 6G
http://www.deepsense6g.net/

Description: Evaluation script for radar aided beam prediction task
Author: Umut Demirhan
Date: 10/29/2021
"""


import torch

from radar_network_models import LeNet_RadarCube, LeNet_RangeAngle, LeNet_RangeVelocity
from radar_preprocessing_torch import range_velocity_map, range_angle_map, radar_cube
from network_functions import test_loop, evaluate_predictions
from dataset import load_radar_data

# Model Name
model_folder = 'type0_split100_batchsize32_rng0_epoch40' #  Radar Cube solution
# model_folder = 'type1_split100_batchsize32_rng0_epoch40' # Range-Velocity solution
# model_folder = 'type2_split100_batchsize32_rng0_epoch40' # Range-Angle solution

model_path = './saved_models/' + model_folder + '/'

# Dataset Files
dataset_dir = r'.\Scenario9\development_dataset' # Development dataset location
csv_file = 'scenario9_dev_test.csv' # Test CSV file

# Solution Type Selection / Automatically extracted from folder name
data_type = int(model_folder.split('_')[0][-1]) # 0: Radar Cube / 1: Range Velocity / 2: Range Angle

# Define set of preprocessing and neural networks for different solutions
preprocessing_functions = [radar_cube, range_velocity_map, range_angle_map]
neural_nets = [LeNet_RadarCube, LeNet_RangeVelocity, LeNet_RangeAngle]

# Load Data
X_test, y_test = load_radar_data(dataset_dir, csv_file, radar_column='unit1_radar_1', label_column='beam_index_1')

# Radar Preprocessing
preprocessing_function = preprocessing_functions[data_type]
X_test = preprocessing_function(X_test)

# PyTorch Tensors
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

# Neural Network Settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = neural_nets[data_type]()
net.to(device)

print('Testing..')
topk = 5

y, y_hat, network_time_per_sample = test_loop(X_test, y_test, net, device, model_path=model_path+'modelbest.pth') # Best model

# This model was trained on flipped power levels..
# To flip the labels in [1, 64], apply the transformation
y = 65 - y

topk_acc_best, beam_dist_best = evaluate_predictions(y, y_hat, k=topk)
print('Best model:')
print('Top-k Accuracy: ' + '-'.join(['%.2f' for i in range(topk)]) % tuple(topk_acc_best*100))
