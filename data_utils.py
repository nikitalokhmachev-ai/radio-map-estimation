import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class MapDataset(torch.utils.data.IterableDataset):
    def __init__(self, pickles, scaler=None):
        super().__init__()
        self.pickles = pickles
        self.scaler = scaler

    def __iter__(self):
        yield from file_path_generator(self.pickles, self.scaler)

def file_path_generator(pickles, scaler):
    for file_path in pickles:
        t_x_points, t_y_points, t_y_masks, t_channel_pows = load_numpy_array(file_path, scaler)
        for i, (t_x_point, t_y_point, t_y_mask, t_channel_pow) in enumerate(zip(t_x_points, t_y_points, t_y_masks, t_channel_pows)):
            yield t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, i


def load_numpy_array(file_path, scaler):
    t_x_points, t_channel_pows, t_y_masks = np.load(file_path, allow_pickle=True)
    t_y_points = t_channel_pows * t_y_masks
    if scaler:
        t_x_mask = t_x_points[:,1,:,:] == 1
        t_x_points[:,0,:,:] = scaler.transform(t_x_points[:,0,:,:]) * t_x_mask
        t_channel_pows = scaler.transform(t_channel_pows)
        t_y_points = scaler.transform(t_y_points)
    return t_x_points, t_y_points, t_y_masks, t_channel_pows
  

class Scaler():
    def __init__(self, scaler='minmax', bounds=(0, 1)):
        self.scaler = scaler
        self.bounds = bounds
        if scaler == 'minmax':
            self.sc = MinMaxScaler(feature_range=self.bounds)
        else:
            self.sc = StandardScaler()
    
    def fit(self, data):
        data = data.flatten().reshape(-1,1)
        self.sc.partial_fit(data)

    def transform(self, data):
        data_shape = data.shape
        data = data.flatten().reshape(-1,1)
        data = self.sc.transform(data)
        data = data.reshape(data_shape)        
        return data
    
    def reverse_transform(self, data):
        data_shape = data.shape
        data = data.flatten().reshape(-1,1)
        data = self.sc.inverse_transform(data)
        data = data.reshape(data_shape)        
        return data

def train_scaler(scaler, pickles):
    gen = file_path_generator(pickles, scaler=None)
    for t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, i in gen:
        scaler.fit(t_channel_pow)