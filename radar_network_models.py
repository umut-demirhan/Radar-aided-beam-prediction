"""
The Wireless Intelligence Lab

DeepSense 6G
http://www.deepsense6g.net/

Description: Pytorch deep neural network model definitions
Author: Umut Demirhan
Date: 10/29/2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class LeNet_RadarCube(nn.Module):
    def __init__(self):
        super(LeNet_RadarCube, self).__init__()
        self.pool = nn.AvgPool2d((2, 2))
        self.pool2 = nn.AvgPool2d((2, 1))
        
        self.conv1 = nn.Conv2d(4, 8, 3, padding='same')
        self.conv2 = nn.Conv2d(8, 16, 3, padding='same')
        self.conv3 = nn.Conv2d(16, 8, 3, padding='same')
        self.conv4 = nn.Conv2d(8, 4, 3, padding='same')
        self.conv5 = nn.Conv2d(4, 2, 3, padding='same')
        self.fc1 = nn.Linear(512, 4*64)
        self.fc2 = nn.Linear(4*64, 2*64)
        self.fc3 = nn.Linear(2*64, 64)

    def forward(self, x):
        
        x = F.relu((self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))
        x = F.relu(self.pool(self.conv5(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet_RangeAngle(nn.Module):
    def __init__(self):
        super(LeNet_RangeAngle, self).__init__()
        self.pool = nn.AvgPool2d((2, 2))
        
        self.conv1 = nn.Conv2d(1, 8, 3, padding='same')
        self.conv2 = nn.Conv2d(8, 16, 3, padding='same')
        self.conv3 = nn.Conv2d(16, 8, 3, padding='same')
        self.conv4 = nn.Conv2d(8, 4, 3, padding='same')
        self.conv5 = nn.Conv2d(4, 2, 3, padding='same')
        self.fc1 = nn.Linear(512, 4*64)
        self.fc2 = nn.Linear(4*64, 2*64)
        self.fc3 = nn.Linear(2*64, 64)

    def forward(self, x):
        
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))
        x = F.relu(self.pool(self.conv5(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class LeNet_RangeVelocity(nn.Module):
    def __init__(self):
        super(LeNet_RangeVelocity, self).__init__()
        self.pool = nn.AvgPool2d((2, 2))
        self.pool2 = nn.AvgPool2d((2, 1))
        
        self.conv1 = nn.Conv2d(1, 8, 3, padding='same')
        self.conv2 = nn.Conv2d(8, 16, 3, padding='same')
        self.conv3 = nn.Conv2d(16, 8, 3, padding='same')
        self.conv4 = nn.Conv2d(8, 4, 3, padding='same')
        self.conv5 = nn.Conv2d(4, 2, 3, padding='same')
        self.fc1 = nn.Linear(512, 4*64)
        self.fc2 = nn.Linear(4*64, 2*64)
        self.fc3 = nn.Linear(2*64, 64)

    def forward(self, x):
        
        x = F.relu((self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))
        x = F.relu(self.pool(self.conv5(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x