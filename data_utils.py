import torch
import torchvision
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
import os
import glob
import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

# Iterable Dataset class for Deep Completion Autoencoders data

class MapDataset(torch.utils.data.IterableDataset):
    def __init__(self, pickles, scaler=None, building_value=None, unsampled_value=None, sampled_value=None):
        super().__init__()
        self.pickles = pickles
        self.scaler = scaler
        self.building_value = building_value
        self.unsampled_value = unsampled_value
        self.sampled_value = sampled_value

    def __iter__(self):
        yield from file_path_generator(self.pickles, self.scaler, self.building_value, self.unsampled_value, self.sampled_value)

def file_path_generator(pickles, scaler, building_value=None, unsampled_value=None, sampled_value=None):
    for file_path in pickles:
        t_x_points, t_y_points, t_y_masks, t_channel_pows = load_numpy_array(
            file_path, scaler, building_value=building_value, unsampled_value=unsampled_value, sampled_value=sampled_value)
        for i, (t_x_point, t_y_point, t_y_mask, t_channel_pow) in enumerate(zip(t_x_points, t_y_points, t_y_masks, t_channel_pows)):
            yield t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, i


def load_numpy_array(file_path, scaler, building_value=None, unsampled_value=None, sampled_value=None):
    t_x_points, t_channel_pows, t_y_masks = np.load(file_path, allow_pickle=True)
    t_y_points = t_channel_pows * t_y_masks

    if scaler:
        t_x_mask = t_x_points[:,1,:,:] == 1
        t_x_points[:,0,:,:] = scaler.transform(t_x_points[:,0,:,:]) * t_x_mask
        t_channel_pows = scaler.transform(t_channel_pows)
        t_y_points = scaler.transform(t_y_points)

    if building_value:
        t_x_points[:,0][t_x_points[:,1] == -1] = building_value
    
    if unsampled_value:
        t_x_points[:,0][t_x_points[:,1] == 0] = unsampled_value

    if sampled_value:
        t_x_points[:,0][t_x_points[:,1] == 1] += sampled_value
    
    return t_x_points, t_y_points, t_y_masks, t_channel_pows
  

class Scaler():
    def __init__(self, scaler='minmax', bounds=(0, 1), min_trunc=None, max_trunc=None):
        self.scaler = scaler
        self.bounds = bounds
        self.min_trunc = min_trunc
        self.max_trunc = max_trunc
        if scaler == 'minmax':
            self.sc = MinMaxScaler(feature_range=self.bounds)
        else:
            self.sc = StandardScaler()
    
    def fit(self, data):
        data = data.flatten().reshape(-1,1)
        self.sc.partial_fit(data)
        if self.min_trunc:
            if self.sc.data_min_ < self.min_trunc:
                self.sc.data_min_ = self.min_trunc
        if self.max_trunc:
            if self.sc.data_max_ > self.max_trunc:
                self.sc.data_max_ = self.max_trunc

    def transform(self, data):
        data_shape = data.shape
        data = data.flatten().reshape(-1,1)
        if self.min_trunc:
            data[data < self.min_trunc] = self.min_trunc
        if self.max_trunc:
            data[data > self.max_trunc] = self.max_trunc
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


# Dataset class for RadioMapSeer data

class RadioMapDataset(Dataset):
  def __init__(self, data_dir):
    super().__init__()
    self.samples = glob.glob(os.path.join(data_dir, '*'))
  
  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    path = self.samples[idx]
    with open(path, 'rb') as f:
      sampled_map, complete_map, building_mask, tx_loc = pickle.load(f)
    return sampled_map, complete_map, building_mask, complete_map, path, tx_loc
  

class RadioMapGenerator(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.map_paths = sorted(glob.glob(os.path.join(data_dir, 'gain', 'DPM', '*')))

    def __len__(self):
        return len(self.map_paths)

    def __getitem__(self, idx):
        # Get path to complete map file
        map_path = self.map_paths[idx]
        map_name = os.path.basename(map_path)

        # Get paths to corresponding building mask and tx location files
        building_name = map_name.split('_')[0]
        building_path = os.path.join(self.data_dir, 'png', 'buildings_complete', f'{building_name}.png')
        tx_path = os.path.join(self.data_dir, 'antenna', f'{building_name}.json')
        tx_name = int(map_name.split('_')[1][:-4])

        # Open complete map, building mask, and tx loc files. 
        complete_map = read_image(self.map_paths[idx])
        building_mask = read_image(building_path)
        with open(tx_path) as f:
            tx_file = json.load(f)
        x_loc, y_loc = tx_file[tx_name]

        # Scale map and mask from 0 - 255 to 0 - 1. Scale tx_file by size of image.
        complete_map = complete_map / 255
        building_mask = building_mask / 255
        x_loc = x_loc / complete_map.shape[2]
        y_loc = y_loc / complete_map.shape[1]
        tx_loc = torch.tensor([x_loc, y_loc])

        # Sample map for model input
        sampled_map, environment_mask = sample_map(sampling_factor=0.1, map_to_sample=complete_map, building_mask=building_mask)

        # Concatenate sampled_map and environment_mask, invert building_mask
        sampled_map = torch.cat((sampled_map, environment_mask))
        building_mask = 1-building_mask

        # from MapDataset class, corresponds to:
        # return t_x_points, t_y_points, t_y_mask, t_channel_pows, filename, i
        # `complete_map` is returned twice for compatibility with previous class. `i` is replaced wtih tx_loc
        return sampled_map, complete_map, environment_mask, complete_map,  map_path, tx_loc
    

def sample_map(sampling_factor, map_to_sample, building_mask):
    """
    This function is copied with some edits from the repository fachu000/deep-autoencoders-cartography 
    https://github.com/fachu000/deep-autoencoders-cartography

    Params:
    sampling_factor: can be
        - fraction between 0 and 1 determining on average the percentage of entries to select
            as samples, i.e  sampling_factor= 0.3 will allow selection of 30 % of total entries of 
            the map that are different from np.nan. 
        - Interval (list or tuple of length 2). The aforementioned fraction is drawn uniformly at random 
            within the interval [sampling_factor[0], sampling_factor[1] ] each time a map is sampled.
    map_to_sample: 1 x Nx x Ny tensor, complete map of radio power, scaled between 0 and 1
    building_mask: 1 x Nx x Ny tensor, binary mask of building locations, each entry is 1 if occupied by building, 0 otherwise

    Returns:
    sampled map: 1 x Nx x Ny tensor
    environment_mask`: 1 x Nx x Ny, each entry is 1 if that grid point is sampled, -1 if occupied by building, 0 otherwise.
    """
    if np.size(sampling_factor) == 1:
        pass
    elif np.size(sampling_factor) == 2:
        sampling_factor = np.round_((sampling_factor[1] - sampling_factor[0]) * np.random.rand() +
                                    sampling_factor[0], decimals=2)
    else:
        Exception("invalid value of v_sampling_factor")
    if sampling_factor == 1:
        sampled_map = map_to_sample.clone()
        environment_mask = torch.ones(map_to_sample.shape) -2 * building_mask
    else:
        flat_map = map_to_sample.clone().flatten()
        flat_buildings = building_mask.clone().flatten()
        flat_environment = torch.ones(flat_map.shape)
        
        building_idx = torch.nonzero(flat_buildings != 0).squeeze()
        free_space_idx = torch.nonzero(flat_buildings == 0).squeeze()
        unsampled_idx = np.random.choice(free_space_idx,
                                        size=int((1 - sampling_factor) * len(free_space_idx)),
                                        replace=False)
        all_unsampled_idx = list(map(int, np.concatenate((building_idx, unsampled_idx),
                                                        axis=0)))

        flat_map[all_unsampled_idx] = 0
        flat_environment[building_idx] = -1
        flat_environment[unsampled_idx] = 0
        sampled_map = torch.reshape(flat_map, map_to_sample.shape)
        environment_mask = torch.reshape(flat_environment, map_to_sample.shape)
    return sampled_map, environment_mask