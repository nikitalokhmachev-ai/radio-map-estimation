from .base import Encoder as BaseEncoder, Decoder as BaseDecoder
from .resnet import Encoder as ResnetEncoder, Decoder as ResnetDecoder
from .unet import Encoder as UNetEncoder, Decoder as UNetDecoder
from .res_unet import Encoder as ResUNetEncoder, Decoder as ResUNetDecoder
from .vae import Encoder as VariationalEncoder, Decoder as VariationalDecoder
from .resnet_vae import Encoder as ResnetVariationalEncoder, Decoder as ResnetVariationalDecoder
from .sparse_base import Encoder as SparseBaseEncoder, Decoder as SparseBaseDecoder
from .sparse_unet import Encoder as SparseUNetEncoder, Decoder as SparseUNetDecoder
from .sparse_base_maxpool import Encoder as SparseBaseEncoder_MaxPool, Decoder as SparseBaseDecoder_MaxPool
from .sparse_unet_maxpool import Encoder as SparseUnetEncoder_MaxPool, Decoder as SparseUNetDecoder_MaxPool
from .autoencoder import Autoencoder

import torch
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class BaseAutoencoder(Autoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = BaseEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = BaseDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)



class ResnetAutoencoder(Autoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = ResnetEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = ResnetDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)



class UNetAutoencoder(Autoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = UNetEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = UNetDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, skip1, skip2, skip3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3)
        return x



class ResUNetAutoencoder(UNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = ResUNetEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = ResUNetDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)



class VariationalAutoencoder(Autoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = VariationalEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = VariationalDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x, x_mu, x_logvar = self.encoder(x)
        x = self.decoder(x)
        return x, x_mu, x_logvar
        
    def fit(self, train_dl, optimizer, epochs=100, loss='mse'):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_dl):
                optimizer.zero_grad()
                t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                t_channel_pow = t_channel_pow.flatten(1).to(device)
                t_y_point_pred, means, logvars = self.forward(t_x_point)
                t_y_point_pred, means, logvars = t_y_point_pred.to(torch.float64), means.to(torch.float64), logvars.to(torch.float64)
                loss_ = torch.nn.functional.mse_loss(t_y_point * t_y_mask, t_y_point_pred * t_y_mask).to(torch.float32)
                if loss == 'rmse':
                    loss_ = torch.sqrt(loss_)
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
                    t_channel_pow = t_channel_pow.flatten(1).to(device).detach().cpu().numpy()
                    _, means, _ = self.encoder(t_x_point)
                    t_y_point_pred = self.decoder(means).detach().cpu().numpy()
                    building_mask = (t_x_point[:,1,:,:].flatten(1) == -1).to(torch.float64).detach().cpu().numpy()
                    loss = (np.linalg.norm((1 - building_mask) * (scaler.reverse_transform(t_channel_pow) - scaler.reverse_transform(t_y_point_pred)), axis=1) ** 2 / np.sum(building_mask == 0, axis=1)).tolist()
                    losses += loss
            
                    print(f'{np.sqrt(np.mean(loss))}')
                    
            return torch.sqrt(torch.Tensor(losses).mean())

class VariationalAutoencoderRandomness(VariationalAutoencoder):

    def evaluate(self, test_dl, scaler):
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_dl):
                    t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                    t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                    t_channel_pow = t_channel_pow.flatten(1).to(device).detach().cpu().numpy()
                    x, _, _ = self.encoder(t_x_point)
                    t_y_point_pred = self.decoder(x).detach().cpu().numpy()
                    building_mask = (t_x_point[:,1,:,:].flatten(1) == -1).to(torch.float64).detach().cpu().numpy()
                    loss = (np.linalg.norm((1 - building_mask) * (scaler.reverse_transform(t_channel_pow) - scaler.reverse_transform(t_y_point_pred)), axis=1) ** 2 / np.sum(building_mask == 0, axis=1)).tolist()
                    losses += loss
            
                    print(f'{np.sqrt(np.mean(loss))}')
                    
            return torch.sqrt(torch.Tensor(losses).mean())



class ResnetVariationalAutoencoder(VariationalAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = ResnetVariationalEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = ResnetVariationalDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        


class SparseBaseAutoencoder(Autoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = SparseBaseEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = SparseBaseDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x, mask):
        x = self.encoder(x, mask)
        x = self.decoder(x)
        return x

    def fit(self, train_dl, optimizer, epochs=100, loss='mse'):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_dl):
                optimizer.zero_grad()
                t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                t_channel_pow = t_channel_pow.flatten(1).to(device)
                mask = (t_x_point[:,1] != 0).to(torch.float32)
                t_y_point_pred = self.forward(t_x_point[:,0].unsqueeze(1), mask.unsqueeze(1)).to(torch.float64)
                loss_ = torch.nn.functional.mse_loss(t_y_point * t_y_mask, t_y_point_pred * t_y_mask).to(torch.float32)
                if loss == 'rmse':
                    loss_ = torch.sqrt(loss_)
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
                    t_channel_pow = t_channel_pow.flatten(1).to(device).detach().cpu().numpy()
                    mask = (t_x_point[:,1] != 0).to(torch.float32)
                    t_y_point_pred = self.forward(t_x_point[:,0].unsqueeze(1), mask.unsqueeze(1)).detach().cpu().numpy()
                    building_mask = (t_x_point[:,1,:,:].flatten(1) == -1).to(torch.float64).detach().cpu().numpy()
                    loss = (np.linalg.norm((1 - building_mask) * (scaler.reverse_transform(t_channel_pow) - scaler.reverse_transform(t_y_point_pred)), axis=1) ** 2 / np.sum(building_mask == 0, axis=1)).tolist()
                    losses += loss
            
                    print(f'{np.sqrt(np.mean(loss))}')
                    
            return torch.sqrt(torch.Tensor(losses).mean())
        


class SparseUNetAutoencoder(SparseBaseAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = SparseUNetEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = SparseUNetDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x, mask):
        x, skip1, skip2, skip3 = self.encoder(x, mask)
        x = self.decoder(x, skip1, skip2, skip3)
        return x
    


class SparseBaseAutoencoder_MaxPool(SparseBaseAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = SparseBaseEncoder_MaxPool(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = SparseBaseDecoder_MaxPool(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)



class SparseUNetAutoencoder_MaxPool(SparseUNetAutoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = SparseUnetEncoder_MaxPool(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = SparseUNetDecoder_MaxPool(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)



class UNetAutoencoder_NoMask(UNetAutoencoder):
    def __init__(self, enc_in=1, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = UNetEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = UNetDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def fit(self, train_dl, optimizer, epochs=100, loss='mse'):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_dl):
                optimizer.zero_grad()
                t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                t_x_point = t_x_point[:,0].unsqueeze(1)
                t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                t_channel_pow = t_channel_pow.flatten(1).to(device)
                t_y_point_pred = self.forward(t_x_point).to(torch.float64)
                loss_ = torch.nn.functional.mse_loss(t_y_point * t_y_mask, t_y_point_pred * t_y_mask).to(torch.float32)
                if loss == 'rmse':
                    loss_ = torch.sqrt(loss_)
                loss_.backward()
                optimizer.step()

                running_loss += loss_.item()        
                print(f'{loss_}, [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1)}')


    def evaluate(self, test_dl, scaler):
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_dl):
                    t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                    building_mask = (t_x_point[:,1,:,:].flatten(1) == -1).to(torch.float64).detach().cpu().numpy()
                    t_x_point = t_x_point[:,0].unsqueeze(1)
                    t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                    t_channel_pow = t_channel_pow.flatten(1).to(device).detach().cpu().numpy()
                    t_y_point_pred = self.forward(t_x_point).detach().cpu().numpy()
                    loss = (np.linalg.norm((1 - building_mask) * (scaler.reverse_transform(t_channel_pow) - scaler.reverse_transform(t_y_point_pred)), axis=1) ** 2 / np.sum(building_mask == 0, axis=1)).tolist()
                    losses += loss
            
                    print(f'{np.sqrt(np.mean(loss))}')
                    
            return torch.sqrt(torch.Tensor(losses).mean())