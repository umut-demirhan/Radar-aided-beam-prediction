"""
The Wireless Intelligence Lab

DeepSense 6G
http://www.deepsense6g.net/

Description: Pytorch network loops
Author: Umut Demirhan
Date: 10/29/2021
"""

import torch
from tqdm import tqdm
import numpy as np
import sys

def train_loop(X_train, y_train, net, optimizer, criterion, device, batch_size=64):
    net.train()
    
    running_acc = 0.0
    running_loss = 0.0
    with tqdm(iterate_minibatches(X_train, y_train, batch_size, shuffle=True), unit=' batch', 
              total=int(np.ceil(X_train.shape[0]/batch_size)), file=sys.stdout, leave=True) as tepoch:
        for batch_x, batch_y in tepoch:
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad() # Make the gradients zero
            batch_y_hat = net(batch_x) # Prediction
            loss = criterion(batch_y_hat, batch_y) # Loss computation
            loss.backward() # Backward step
            optimizer.step() # Update coefficients
            
            predictions = batch_y_hat.argmax(dim=1, keepdim=True).squeeze()
            
            batch_correct = (predictions == batch_y).sum().item()
            running_acc += batch_correct
            batch_loss = loss.item()
            running_loss += batch_loss
            
            tepoch.set_postfix(loss=batch_loss, accuracy=100. * batch_correct/batch_size)
            
        curr_acc = 100. * running_acc/len(X_train)
        curr_loss = running_loss/np.ceil(X_train.shape[0]/batch_size)

    return curr_loss, curr_acc
    
def eval_loop(X_val, y_val, net, criterion, device, batch_size=64):
    net.eval()
    
    running_acc = 0.0
    running_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in iterate_minibatches(X_val, y_val, batch_size, shuffle=True):
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            batch_y_hat = net(batch_x) # Prediction
    
            predictions = batch_y_hat.argmax(dim=1, keepdim=True).squeeze()
            running_acc += (predictions == batch_y).sum().item()
            running_loss += criterion(batch_y_hat, batch_y).item()
        
    
    curr_acc = 100. * running_acc/len(X_val)
    curr_loss = running_loss/np.ceil(X_val.shape[0]/batch_size)
    print('Validation: [accuracy=%.2f, loss=%.4f]' % (curr_acc, curr_loss), flush=True)
    
    return curr_loss, curr_acc
    
def test_loop(X_test, y_test, net, device, model_path):
    # Network Setup
    net.load_state_dict(torch.load(model_path))
    net.eval()
    
    # Timers
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_time = 0
    
    #GPU-WARM-UP
    data_shape_single = list(X_test.shape)
    data_shape_single[0] = 1
    data_shape_single = tuple(data_shape_single)
    with torch.no_grad():
        for _ in range(10):
            dummy_input = torch.randn(data_shape_single, dtype=torch.float, device=device)
            out = net(dummy_input)
    
    
    
    y = -1*torch.ones(len(X_test))
    y_hat = -1*torch.ones((len(X_test), out.shape[1])) #Top 5
    
    cur_ind = 0
    
    # Test
    with torch.no_grad():
        for batch_x, batch_y in iterate_minibatches(X_test, y_test, batchsize=1, shuffle=False):
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            #### Measure Inference Duration ####
            starter.record()
            
            batch_y_hat = net(batch_x) # Prediction
            
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time
            #####################################
            
            ###### Save Top-k Predictions #######
            next_ind = cur_ind + batch_x.shape[0]
            y[cur_ind:next_ind] = batch_y
            y_hat[cur_ind:next_ind, :] = batch_y_hat
            cur_ind = next_ind
            #####################################

    network_time_per_sample = total_time / len(X_test)
    
    return y, y_hat, network_time_per_sample
    

def evaluate_predictions(y, y_hat, k):
    topk_pred = torch.topk(y_hat, k=k).indices
    topk_acc = np.zeros(k)
    for i in range(k):
        topk_acc[i] = torch.mean((y == topk_pred[:, i])*1.0)
    topk_acc = np.cumsum(topk_acc)
    
    beam_dist = torch.mean(torch.abs(y - topk_pred[:, 0]))
    
    return topk_acc, beam_dist

def iterate_minibatches(X, y, batchsize, shuffle=False):
    
    data_len = X.shape[0]
    indices = np.arange(data_len)
    if shuffle:
        np.random.shuffle(indices)
        
    for start_idx in range(0, data_len, batchsize):
        end_idx = min(start_idx + batchsize, data_len)
        excerpt = indices[start_idx:end_idx]
        yield X[excerpt], y[excerpt]
        