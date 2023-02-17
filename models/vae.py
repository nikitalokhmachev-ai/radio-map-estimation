import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, enc_in, enc_out, n_dim, leaky_relu_alpha=0.3):
        super(Encoder, self).__init__()
        self.conv2d = nn.Conv2d(enc_in, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_1 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_2 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.average_pooling2d = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv2d_3 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_4 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_5 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.average_pooling2d_1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv2d_6 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_7 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_8 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.average_pooling2d_2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.mu = nn.Conv2d(n_dim, enc_out, kernel_size=(3, 3), padding='same')
        self.logvar = nn.Conv2d(n_dim, enc_out, kernel_size=(3, 3), padding='same')

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_alpha)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.leaky_relu(self.conv2d(x))
        
        x = self.leaky_relu(self.conv2d_1(x))
        x = self.leaky_relu(self.conv2d_2(x))
        x = self.average_pooling2d(x)
        
        x = self.leaky_relu(self.conv2d_3(x))
        x = self.leaky_relu(self.conv2d_4(x))
        x = self.leaky_relu(self.conv2d_5(x))
        x = self.average_pooling2d_1(x)
        
        x = self.leaky_relu(self.conv2d_6(x))
        x = self.leaky_relu(self.conv2d_7(x))
        x = self.leaky_relu(self.conv2d_8(x))
        x = self.average_pooling2d_2(x)
        x_mu = self.mu(x)
        x_logvar = self.logvar(x)
        x = self.reparameterize(x_mu, x_logvar)
        x = self.leaky_relu(x)
        return x
    

#@title Decoder
class Decoder(nn.Module):
    def db_to_natural(self, x):
        return 10 ** (x / 10)
      
    def __init__(self, dec_in, dec_out, n_dim, leaky_relu_alpha=0.3):
        super(Decoder, self).__init__()

        self.conv2d_transpose = nn.ConvTranspose2d(dec_in, dec_in, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_1 = nn.ConvTranspose2d(dec_in, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_2 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_3 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_4 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_5 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_6 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_7 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_8 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3,3), stride=1, padding=1)
        self.conv2d_transpose_9 = nn.ConvTranspose2d(n_dim, dec_out, kernel_size=(3,3), stride=1, padding=1)

        self.up_sampling2d = nn.Upsample(scale_factor=2)
        self.up_sampling2d_1 = nn.Upsample(scale_factor=2)
        self.up_sampling2d_2 = nn.Upsample(scale_factor=2)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_alpha)
        self.bases = torch.tensor([[1]], dtype=torch.float32).to(device)
        self.log_10 = torch.log(torch.tensor([10])).to(device)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                

    def forward(self, x):
        x = self.leaky_relu(self.conv2d_transpose(x))
        x = self.up_sampling2d(x)
        x = self.leaky_relu(self.conv2d_transpose_1(x))
        x = self.leaky_relu(self.conv2d_transpose_2(x))
        x = self.leaky_relu(self.conv2d_transpose_3(x))
        x = self.up_sampling2d_1(x)
        x = self.leaky_relu(self.conv2d_transpose_4(x))
        x = self.leaky_relu(self.conv2d_transpose_5(x))
        x = self.leaky_relu(self.conv2d_transpose_6(x))
        x = self.up_sampling2d_2(x)
        x = self.leaky_relu(self.conv2d_transpose_7(x))
        x = self.leaky_relu(self.conv2d_transpose_8(x))
        x = self.leaky_relu(self.conv2d_transpose_9(x))
        x = torch.flatten(x, start_dim=1)

        return x