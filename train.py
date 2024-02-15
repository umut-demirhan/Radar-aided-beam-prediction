"""
The Wireless Intelligence Lab

DeepSense 6G
http://www.deepsense6g.net/

Description: Training script for radar aided beam prediction task
            The script does not reproduce the results exactly as provided in the paper.
            It is provided as a reference for training the models.
Author: Umut Demirhan
Date: 02/14/2024
"""

import os
import torch
import numpy as np

from radar_network_models import LeNet_RadarCube, LeNet_RangeAngle, LeNet_RangeVelocity
from radar_preprocessing_torch import range_velocity_map, range_angle_map, radar_cube
from network_functions import test_loop, train_loop, eval_loop, evaluate_predictions
from dataset import load_radar_data

# Training parameters
rng_seed = 0
np.random.seed(rng_seed)
torch.manual_seed(rng_seed)
torch.backends.cudnn.deterministic = True

num_epochs = 40
batch_size = 32
data_type = 0 # 0: Radar Cube / 1: Range-Velocity / 2: Range-Angle

model_name = ['RadarCube', 'RangeVelocity', 'RangeAngle'][data_type] + '_batchsize%i' % batch_size + '_epoch%i' % num_epochs
save_path = './models/%s' % model_name

# Dataset Files
dataset_dir = r'C:\Users\Umt\Desktop\Scenario9\development_dataset' # Development dataset location
train_csv_file = 'scenario9_dev_train.csv' # Train CSV file
test_csv_file = 'scenario9_dev_test.csv' # Test CSV file

# Define set of preprocessing and neural networks for different solutions
preprocessing_functions = [radar_cube, range_velocity_map, range_angle_map]
neural_nets = [LeNet_RadarCube, LeNet_RangeVelocity, LeNet_RangeAngle]

# Load Data
X_train, y_train = load_radar_data(dataset_dir, train_csv_file, radar_column='unit1_radar_1', label_column='beam_index_1')
X_test, y_test = load_radar_data(dataset_dir, test_csv_file, radar_column='unit1_radar_1', label_column='beam_index_1')

# Radar Preprocessing
preprocessing_function = preprocessing_functions[data_type]
X_train = preprocessing_function(X_train)
X_test = preprocessing_function(X_test)

# PyTorch Tensors
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

# Neural Network Settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = neural_nets[data_type]()

net.to(device)
criterion = torch.nn.CrossEntropyLoss() # Mean for training
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(num_epochs):
    print('Epoch %i/%i'%(epoch+1, num_epochs))
    # Training Loop
    train_loop(X_train, y_train, net, optimizer, criterion, device, batch_size=batch_size)
    eval_loop(X_test, y_test, net, criterion, device, batch_size=256)
    scheduler.step()

if not os.path.exists(save_path):
    os.makedirs(save_path)
model_path = os.path.join(save_path, 'model.pth')
torch.save(net.state_dict(), model_path)
print('Finished Training')

y, y_hat, network_time_per_sample = test_loop(X_test, y_test, net, device, model_path=None) # Best model

topk = 5
topk_acc_best, beam_dist_best = evaluate_predictions(y, y_hat, k=topk)
print('Best model:')
print('Top-k Accuracy: ' + '-'.join(['%.2f' for i in range(topk)]) % tuple(topk_acc_best*100))