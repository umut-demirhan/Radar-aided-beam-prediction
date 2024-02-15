# -*- coding: utf-8 -*-
"""
The Wireless Intelligence Lab

DeepSense 6G
http://www.deepsense6g.net/

Description: Training script for radar aided beam prediction task
Author: Umut Demirhan
Date: 02/15/2024
"""


import os
import torch
import numpy as np

from radar_network_models import LeNet_RadarCube, LeNet_RangeAngle, LeNet_RangeVelocity
from radar_preprocessing_torch import range_velocity_map, range_angle_map, radar_cube
from network_functions import train_loop, eval_loop, test_loop, evaluate_predictions
from dataset import load_radar_data

# Dataset Files
root_dir = r'.\Scenario9\development_dataset'
csv_files = ['scenario9_dev_%s.csv'%s for s in ['train', 'val', 'test']]

# Solution Type Selection
data_type = 0 # 0: Radar Cube / 1: Range Velocity / 2: Range Angle
print('Model/Data Type: %s' % ['Radar Cube', 'Range Velocity', 'Range Angle'][data_type])

# Training Settings
batch_size = 32
num_epoch = 40
learning_rate = 1e-3
rng_seed = 0

# Define set of preprocessing and neural networks for different solutions
preprocessing_functions = [radar_cube, range_velocity_map, range_angle_map]
neural_nets = [LeNet_RadarCube, LeNet_RangeVelocity, LeNet_RangeAngle]

#%% Data Prep

# Load Data
X_train, y_train = load_radar_data(root_dir, csv_files[0], radar_column='unit1_radar_1', label_column='beam_index_1')
X_val, y_val = load_radar_data(root_dir, csv_files[1], radar_column='unit1_radar_1', label_column='beam_index_1')
X_test, y_test = load_radar_data(root_dir, csv_files[2], radar_column='unit1_radar_1', label_column='beam_index_1')

# Radar Preprocessing
preprocessing_function = preprocessing_functions[data_type]
X_train = preprocessing_function(X_train)
X_val = preprocessing_function(X_val)
X_test = preprocessing_function(X_test)

# PyTorch Tensors
X_train = torch.from_numpy(X_train)
X_val = torch.from_numpy(X_val)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_val = torch.from_numpy(y_val)
y_test = torch.from_numpy(y_test)

#%% Training

# Model Save Folder
folder_name = 'type%i_batchsize%i_rng%i_epoch%i' %(data_type, batch_size, rng_seed, num_epoch)
models_directory = os.path.abspath('./saved_models/')
if not os.path.exists(models_directory):
    os.makedirs(models_directory)
c = 0
while os.path.exists(os.path.join(models_directory, folder_name + '_v%i'%c, '')):
    c += 1
model_directory = os.path.join(models_directory, folder_name + '_v%i'%c, '')
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
print('Saving the models to %s' % models_directory)

# Reproducibility
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)
torch.backends.cudnn.deterministic = True

# PyTorch GPU/CPU selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Neural Network
net = neural_nets[data_type]()

# Training Settings
net.to(device)
criterion = torch.nn.CrossEntropyLoss() # Training Criterion
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4) # Optimizer
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_loss = np.zeros((num_epoch))
train_acc = np.zeros((num_epoch))
val_loss = np.zeros((num_epoch))
val_acc = np.zeros((num_epoch))

# Epochs
for epoch in range(num_epoch):
    print('Epoch %i/%i:'%(epoch+1, num_epoch), flush=True)
    
    train_loss[epoch], train_acc[epoch] = train_loop(X_train, y_train, net, optimizer, criterion, device, batch_size=batch_size)
    val_loss[epoch], val_acc[epoch] = eval_loop(X_val, y_val, net, criterion, device, batch_size=batch_size)
    
    # Save the best model
    if val_loss[epoch] <= np.min(val_loss[:epoch] if epoch>0 else val_loss[epoch]):
        #print('Saving model..')
        torch.save(net.state_dict(), os.path.join(model_directory, 'model_best.pth'))
        
    scheduler.step()

torch.save(net.state_dict(), os.path.join(model_directory, 'model_final.pth'))

print('Finished Training')

#%% Test

print('Testing..')
topk=5

y, y_hat, network_time_per_sample = test_loop(X_test, y_test, net, device, model_path=os.path.join(model_directory, 'model_best.pth')) # Best model
topk_acc_best, beam_dist_best = evaluate_predictions(y, y_hat, k=topk)
print('Best model:')
print('Top-k Accuracy: ' + '-'.join(['%.2f' for i in range(topk)]) % tuple(topk_acc_best*100))
print('Beam distance: %.2f' % beam_dist_best)


y_final, y_hat_final, network_time_per_sample_final = test_loop(X_test, y_test, net, device, model_path=os.path.join(model_directory, 'model_final.pth')) # Last Epoch
topk_acc_final, beam_dist_final = evaluate_predictions(y_final, y_hat_final, k=topk)
print('Final model:')
print('Top-k Accuracy: ' + '-'.join(['%.2f' for i in range(topk)]) % tuple(topk_acc_final*100))
print('Beam distance: %.2f' % beam_dist_final)