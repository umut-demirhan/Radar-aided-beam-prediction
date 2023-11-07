import torch
import numpy as np
from dataset import load_radar_data
import radar_preprocessing_torch as radar_preprocessing
# import radar_preprocessing as radar_preprocessing
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    root_dir = r'C:\Users\umt\Desktop\scenario9\development_dataset'
    
    csv_file = [r'scenario9_dev_%s.csv' % dataset_type for dataset_type in ['train', 'val', 'test']]
    
    data_type = 2 # 0: Radar Cube, 1: Range-Velocity, 2: Range-Angle
    
    X_train, y_train = load_radar_data(root_dir, csv_file[0], radar_column='unit1_radar_1', label_column='beam_index_1')
    X_val, y_val = load_radar_data(root_dir, csv_file[1], radar_column='unit1_radar_1', label_column='beam_index_1')
    X_test, y_test = load_radar_data(root_dir, csv_file[2], radar_column='unit1_radar_1', label_column='beam_index_1')
    
    preprocessing_types = ['radar_cube', 'range_velocity_map', 'range_angle_map', 'range_angle_map']
    preprocessing_types_fft = [None, None, 64, 4]
    preprocessing_fn = getattr(radar_preprocessing, preprocessing_types[data_type])
    
    X_train = preprocessing_fn(X_train, fft_size=preprocessing_types_fft[data_type])
    X_val = preprocessing_fn(X_val, fft_size=preprocessing_types_fft[data_type])
    X_test = preprocessing_fn(X_test, fft_size=preprocessing_types_fft[data_type])
    
    X_train = torch.from_numpy(X_train)
    X_val = torch.from_numpy(X_val)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)
    y_test = torch.from_numpy(y_test)
        
    # # Normalization
    # max_val = torch.max(train_dataset.data['x'])
    # min_val = torch.min(train_dataset.data['x'])
    # train_dataset.data['x'] = (train_dataset.data['x']-min_val) / (max_val - min_val)
    # test_dataset.data['x'] = (test_dataset.data['x']-min_val) / (max_val - min_val)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    num_training_data = X_train.shape[0]
    num_test_data = X_test.shape[0]
    
    #%%
    number_of_beams = 64
    
    v_train = torch.zeros((num_training_data, 1), dtype=torch.long, device=device)
    for i in range(num_training_data):
        v_train[i] = X_train[i, 0, :, :].argmax()
    
    #plt.hist((64-np.unravel_index(v_train.cpu().numpy(), (256, 64))[1]).squeeze() - train_dataset.data['y'].cpu().numpy(), 64)
    xmax = int(y_train.max())
    xmin = int(y_train.min())
    
    extent = xmin, xmax, 0, 1024
    
    # Look-up Table Generation
    image = -1*torch.ones((np.prod(X_test.shape[1:]), number_of_beams), dtype=torch.long, device=device)
    
    for i in range(num_training_data):
        image[int(v_train[i]),y_train[i]] += 1
        
    plt.imshow(image.cpu().numpy(), extent = extent)

    my_map_values, my_map = image.max(axis=1)
    my_map[(my_map_values == -1)] = -1 # If there is no available data don't make a prediction
    
    pred_train= my_map[v_train]
    accuracy_train = (y_train.reshape(-1, 1) == pred_train).float().mean().cpu().item()
    print('Training Accuracy: %.2f%%'%(accuracy_train*100))
    
    v_test = torch.zeros((num_test_data, 5), dtype=torch.long, device=device)
    pred_test = torch.zeros((num_test_data, 1), dtype=torch.long, device=device)
    
    
    for i in range(num_test_data):
        v_test[i, 0] = X_test[i, 0, :, :].argmax()
        pred_test[i] = my_map[v_test[i, 0]]
        
        j=1
        jc=0
        cur_data = X_test[i, 0, :, :].flatten().sort(descending=True)[1]
        while j<5 and jc<X_test.shape[0]:
            if not torch.sum(my_map[v_test[i, :]]==my_map[cur_data[jc]]):
                v_test[i, j]=cur_data[jc]
                j +=1
            jc += 1
        
    
    pred_test = my_map[v_test]
    accuracy_test = (y_test.reshape(-1, 1) == pred_test).float().mean(dim=0).cpu()
    top_k_acc = accuracy_test.cumsum(0)
    
    print('Top-k Test Accuracy: ', end='')
    print(top_k_acc.numpy())
    
    results = {}
    results['Method'] = 'lookup_table'
    results['Training Accuracy'] = accuracy_train
    results['Top-k Accuracy'] = str(top_k_acc.numpy())
    results['Number of Parameters'] =  np.prod(my_map.shape)
    print(results)