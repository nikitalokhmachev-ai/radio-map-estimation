from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    
    def step(self, batch, optimizer, train=True):
        with torch.set_grad_enabled(train):
            t_x_point, t_y_point, t_y_mask, _, _, tx_loc = batch
            t_x_point, t_y_point = t_x_point.to(torch.float32).to(device), t_y_point.to(torch.float32).to(device)
            t_y_mask, tx_loc = t_y_mask.to(torch.float32).to(device), tx_loc.to(torch.float32).to(device)

            t_y_point_pred = self.forward(t_x_point).to(torch.float32)

            loss_ = nn.functional.mse_loss(t_y_point * t_y_mask, t_y_point_pred * t_y_mask).to(torch.float32)
            
            if train:
                loss_.backward()
                optimizer.step()
                optimizer.zero_grad()

        return loss_
    
    def fit(self, train_dl, optimizer, epochs=100, save_model_epochs=25, save_model_dir ='/content'):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, batch in enumerate(train_dl):
                loss_ = self.step(batch, optimizer, train=True)
                running_loss += loss_.detach().item()
                print(f'{loss_}, [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1)}')

            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                filepath = os.path.join(save_model_dir, f'epoch_{epoch}.pth')
                self.save_model(filepath)

        return running_loss / (i+1)
    

    def fit_wandb(self, train_dl, test_dl, optimizer, project_name, run_name, epochs=100, save_model_epochs=25, save_model_dir='/content'):
        import wandb
        wandb.init(project=project_name, name=run_name)
        for epoch in range(epochs):
            train_running_loss = 0.0
            for i, batch in enumerate(train_dl):
                loss = self.step(batch, optimizer, train=True)
                train_running_loss += loss.detach().item()
                train_loss = train_running_loss/(i+1)
                print(f'{loss}, [{epoch + 1}, {i + 1:5d}] loss: {train_loss}')
                    
            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                filepath = os.path.join(save_model_dir, f'epoch_{epoch}.pth')
                self.save_model(filepath)

            test_running_loss = 0.0
            for i, batch in enumerate(test_dl):
                loss = self.step(batch, optimizer, train=False)
                test_running_loss += loss.detach().item()
                test_loss = test_running_loss/(i+1)
                print(f'{loss}, [{epoch + 1}, {i + 1:5d}] loss: {test_loss}')
                
            wandb.log({'train_loss': train_loss, 'test_loss': test_loss})


    def evaluate(self, test_dl, scaler, dB_max=-47.84, dB_min=-147, no_scale=False):
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_dl):
                    t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                    t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                    t_channel_pow = t_channel_pow.flatten(1).to(device).detach().cpu().numpy()
                    t_y_point_pred = self.forward(t_x_point).flatten(1).detach().cpu().numpy()
                    building_mask = (t_x_point[:,1,:,:].flatten(1) == -1).to(torch.float32).detach().cpu().numpy()
                    if scaler:
                        loss = (np.linalg.norm(
                            (1 - building_mask) * (scaler.reverse_transform(t_channel_pow) - scaler.reverse_transform(t_y_point_pred)), axis=1) ** 2 
                            / np.sum(building_mask == 0, axis=1)).tolist()
                    else:
                        if no_scale==True:
                            loss = (np.linalg.norm(
                                (1 - building_mask) * (t_channel_pow - t_y_point_pred), axis=1) ** 2 
                                / np.sum(building_mask == 0, axis=1)).tolist()
                        else:
                            loss = (np.linalg.norm(
                                (1 - building_mask) * (self.scale_to_dB(t_channel_pow, dB_max, dB_min) - self.scale_to_dB(t_y_point_pred, dB_max, dB_min)), axis=1) ** 2 
                                / np.sum(building_mask == 0, axis=1)).tolist()
                    losses += loss
                    print(f'{np.sqrt(np.mean(loss))}')
                    
            return torch.sqrt(torch.Tensor(losses).mean())
        

    def scale_to_dB(self, value, dB_max, dB_min):
        range_dB = dB_max - dB_min
        dB = value * range_dB + dB_min
        return dB
    

    def save_model(self, out_path):
        torch.save(self, out_path)



class UNet_V2(nn.Module):
    '''
    This UNet is the most like the PIMRC model. It changes Upsampling to ConvTranspose2d, LeakyRelu to Relu (or Sigmoid), 
    in the last layer), adds BatchNorm in the Convolution Blocks, adds a single 1x1 Convolution on the last layer to get
    the right output shape, and changes the dimensionality of the first decoder block to go from latent_channels to features
    instead of latent_channels to latent_channels. It adds roughly 20,000 weights (mostly due to the addition of the Conv
    Transpose2d layers), but this is still smaller than ResUNetConcat. 

    These are variables we don't plan to change / experiment with, so we're checking first to see if they meaningfully
    impact performance when the variables we do mean to change (number of filters at each layer, latent space size) are
    still the same as in the PIMRC UNet.

    We also change the UNet._block so that it includes 3 Convolutions (like the PIMRC UNet) and includes a Bias term, and
    change MaxPool2d to AvgPool2d, again to match PIMRC.
    '''

    def __init__(self, in_channels=2, latent_channels=4, out_channels=1, features=27):
        # TODO: Figure out the best way to call super().__init__() here. 
        '''Right now I've copied all code from UNet, so I don't need to inherit from it. If I inherit from UNet, I can call
        super(UNet_V2, self).__init__(), but that initializes UNet and adds a bunch of parameters I don't need/use. I can
        also call super(UNet, self).__init__(), which seems to just initialize UNet's parent, i.e. nn.Module, which is what
        I want, but I'm nervous there might be some other effects I'm missing there.'''
        super().__init__()

        self.encoder1 = UNet_V2._block(in_channels, features, name="enc1")
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet_V2._block(features, features, name="enc2")
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet_V2._block(features, features, name="enc3")
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder4 = nn.Conv2d(features, latent_channels, kernel_size=3, padding=1)
        # encoder4 (a single Convolution Layer) takes the place of "bottleneck" but serves same purpose

        # convolve once (with a single Convolution Layer) before upsampling / deconvoluting
        self.decoder4 = nn.Conv2d(latent_channels, features, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(features, features, kernel_size=2, stride=2)
        self.decoder3 = UNet_V2._block((features) * 2, features, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features, features, kernel_size=2, stride=2)
        self.decoder2 = UNet_V2._block((features) * 2, features, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features, features, kernel_size=2, stride=2)
        self.decoder1 = UNet_V2._block(features * 2, features, name="dec1")

        # Here is one extra 1x1 Convolution to change the output into the right shape
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        dec4 = self.decoder4(enc4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (
                        name + "conv3",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm3", nn.BatchNorm2d(num_features=features)),
                    (name + "relu3", nn.ReLU(inplace=True)),
                ]
            )
        )
    
    def step(self, batch, optimizer, train=True):
        with torch.set_grad_enabled(train):
            t_x_point, t_y_point, t_y_mask, _, _, tx_loc = batch
            t_x_point, t_y_point = t_x_point.to(torch.float32).to(device), t_y_point.to(torch.float32).to(device)
            t_y_mask, tx_loc = t_y_mask.to(torch.float32).to(device), tx_loc.to(torch.float32).to(device)

            t_y_point_pred = self.forward(t_x_point).to(torch.float32)

            loss_ = nn.functional.mse_loss(t_y_point * t_y_mask, t_y_point_pred * t_y_mask).to(torch.float32)
            
            if train:
                loss_.backward()
                optimizer.step()
                optimizer.zero_grad()

        return loss_
    
    def fit(self, train_dl, optimizer, epochs=100, save_model_epochs=25, save_model_dir ='/content'):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, batch in enumerate(train_dl):
                loss_ = self.step(batch, optimizer, train=True)
                running_loss += loss_.detach().item()
                print(f'{loss_}, [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1)}')

            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                filepath = os.path.join(save_model_dir, f'epoch_{epoch}.pth')
                self.save_model(filepath)

        return running_loss / (i+1)
    

    def fit_wandb(self, train_dl, test_dl, optimizer, project_name, run_name, epochs=100, save_model_epochs=25, save_model_dir='/content'):
        import wandb
        wandb.init(project=project_name, name=run_name)
        for epoch in range(epochs):
            train_running_loss = 0.0
            for i, batch in enumerate(train_dl):
                loss = self.step(batch, optimizer, train=True)
                train_running_loss += loss.detach().item()
                train_loss = train_running_loss/(i+1)
                print(f'{loss}, [{epoch + 1}, {i + 1:5d}] loss: {train_loss}')
                    
            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                filepath = os.path.join(save_model_dir, f'epoch_{epoch}.pth')
                self.save_model(filepath)

            if train_loss > 2:
                wandb.alert(
                    title="Train Loss",
                    text=f"Train Loss on epochs {epoch} equal to {train_loss}")

            test_running_loss = 0.0
            for i, batch in enumerate(test_dl):
                loss = self.step(batch, optimizer, train=False)
                test_running_loss += loss.detach().item()
                test_loss = test_running_loss/(i+1)
                print(f'{loss}, [{epoch + 1}, {i + 1:5d}] loss: {test_loss}')

            if test_loss > 2:
                wandb.alert(
                    title="Test Loss",
                    text=f"Test Loss on epochs {epoch} equal to {train_loss}")
                
            wandb.log({'train_loss': train_loss, 'test_loss': test_loss})


    def evaluate(self, test_dl, scaler, dB_max=-47.84, dB_min=-147, no_scale=False):
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_dl):
                    t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                    t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                    t_channel_pow = t_channel_pow.flatten(1).to(device).detach().cpu().numpy()
                    t_y_point_pred = self.forward(t_x_point).flatten(1).detach().cpu().numpy()
                    building_mask = (t_x_point[:,1,:,:].flatten(1) == -1).to(torch.float32).detach().cpu().numpy()
                    if scaler:
                        loss = (np.linalg.norm(
                            (1 - building_mask) * (scaler.reverse_transform(t_channel_pow) - scaler.reverse_transform(t_y_point_pred)), axis=1) ** 2 
                            / np.sum(building_mask == 0, axis=1)).tolist()
                    else:
                        if no_scale==True:
                            loss = (np.linalg.norm(
                                (1 - building_mask) * (t_channel_pow - t_y_point_pred), axis=1) ** 2 
                                / np.sum(building_mask == 0, axis=1)).tolist()
                        else:
                            loss = (np.linalg.norm(
                                (1 - building_mask) * (self.scale_to_dB(t_channel_pow, dB_max, dB_min) - self.scale_to_dB(t_y_point_pred, dB_max, dB_min)), axis=1) ** 2 
                                / np.sum(building_mask == 0, axis=1)).tolist()
                    losses += loss
                    print(f'{np.sqrt(np.mean(loss))}')
                    
            return torch.sqrt(torch.Tensor(losses).mean())
        

    def scale_to_dB(self, value, dB_max, dB_min):
        range_dB = dB_max - dB_min
        dB = value * range_dB + dB_min
        return dB
    

    def save_model(self, out_path):
        torch.save(self, out_path)