"""
The Wireless Intelligence Lab

DeepSense 6G
http://www.deepsense6g.net/

Description: Preprocessing functions for radar data
Author: Umut Demirhan
Date: 10/29/2021
"""

import numpy as np
from tqdm import tqdm
import torch

def range_velocity_map(data):
    size = list(data.shape)
    size[1] = 1
    new_data = np.zeros(tuple(size), dtype=np.float32)
    for i in tqdm(range(data.shape[0])):
        new_data[i, 0, :, :] = range_velocity_map_single(data[i])
    return new_data
    
def range_angle_map(data, fft_size=64):
    size = list(data.shape)
    size[1] = 1
    size[3] = size[2]
    size[2] = fft_size
    new_data = np.zeros(tuple(size), dtype=np.float32)
    for i in tqdm(range(data.shape[0])):
        new_data[i, 0, :, :] = range_angle_map_single(data[i], fft_size)
    new_data = new_data.transpose([0, 1, 3, 2])
    return new_data

def radar_cube(data, fft_size=4):
    size = list(data.shape)
    size[1] = fft_size
    new_data = np.zeros(tuple(size), dtype=np.float32)
    for i in tqdm(range(data.shape[0])):
        new_data[i, :, :, :] = radar_cube_single(data[i], fft_size)
    return new_data
    
def range_velocity_map_single(data):
    data = torch.from_numpy(data)
    data = torch.fft.fft(data, axis=1) # Range FFT
    data = torch.fft.fft(data, axis=2) # Velocity FFT
    data = torch.abs(data).sum(axis=0) # Sum over antennas
    return data

def range_angle_map_single(data, fft_size = 64):
    data = torch.from_numpy(data)
    data = torch.fft.fft(data, axis=1) # Range FFT
    data -= torch.mean(data, axis=2, keepdims=True)
    data = torch.fft.fft(data, n=fft_size, axis=0) # Angle FFT
    data = torch.abs(data).sum(axis=2) # Sum over velocity
    return data

def radar_cube_single(data, fft_size = 4):
    data = torch.from_numpy(data)
    data = torch.fft.fft(data, axis=1) # Range FFT
    data = torch.fft.fft(data, axis=2) # Velocity FFT
    data = torch.fft.fft(data, n=fft_size, axis=0) # Angle FFT
    data = torch.abs(data)
    return data