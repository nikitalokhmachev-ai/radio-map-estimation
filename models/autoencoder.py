import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = None#Encoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = None#Decoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
     def fit(self, train_dl, optimizer, epochs=100, loss='mse', means=None, logvars=None):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_dl):
                optimizer.zero_grad()
                t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                t_channel_pow = t_channel_pow.flatten(1).to(device)
                t_y_point_pred = self.forward(t_x_point).to(torch.float64)
<<<<<<< HEAD
                loss_ = torch.nn.functional.mse_loss(t_y_point, t_y_point_pred * t_y_mask).to(torch.float32)
=======
                loss = torch.nn.functional.mse_loss(t_y_point * t_y_mask, t_y_point_pred * t_y_mask).to(torch.float32)
>>>>>>> a8e4e04a99c43de54e446d13b05e4a6620b1f4f8
                if loss == 'rmse':
                    loss_ = torch.sqrt(loss)
                if means is not None and logvars is not None:
                    kl_loss = -0.5 * torch.sum(1 + logvars - means.pow(2) - logvars.exp(), dim=1) #Figure out the shapes
                    kl_loss = torch.mean(kl_loss)
                    loss_ += kl_loss
                loss_.backward()
                optimizer.step()

                running_loss += loss_.item()        
                print(f'{loss_}, [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1)}')


    def evaluate(self, test_dl, scaler):
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_dl):
                    t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                    t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                    t_channel_pow = t_channel_pow.flatten(1).to(device)
                    t_y_point_pred = self.forward(t_x_point)
                    building_mask = (t_x_point[:,1,:,:].flatten(1) == -1).to(torch.float64)
                    loss = (torch.norm((1 - building_mask) * (scaler.reverse_transform(t_channel_pow) - scaler.reverse_transform(t_y_point_pred)), dim=1) ** 2 / torch.sum(building_mask == 0, axis=1)).detach().cpu().tolist()
                    losses += loss
            
                    print(f'{np.sqrt(np.mean(loss))}')
                    
            return torch.sqrt(torch.Tensor(losses).mean())
        

    def save_model(self, out_path):
        torch.save(self, out_path)