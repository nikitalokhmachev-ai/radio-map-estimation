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

            loss_ = torch.nn.functional.mse_loss(t_y_point * t_y_mask, t_y_point_pred * t_y_mask).to(torch.float32)
            
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
                    t_x_point, t_y_point, t_y_mask, _, _, tx_loc = data
                    t_x_point, t_y_point = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device)
                    t_y_mask = t_y_mask.flatten(1).to(device)
                    t_y_point_pred = self.forward(t_x_point).detach().cpu().numpy()
                    if scaler:
                        loss_ = (np.linalg.norm(
                            t_y_mask * (scaler.reverse_transform(t_y_point) - scaler.reverse_transform(t_y_point_pred)), axis=1) ** 2 
                            / np.sum(t_y_mask, axis=1)).tolist()
                    else:
                        if no_scale==True:
                            loss_ = (np.linalg.norm(
                                t_y_mask * (t_y_point - t_y_point_pred), axis=1) ** 2 
                                / np.sum(t_y_mask, axis=1)).tolist()
                        else:
                            loss_ = (np.linalg.norm(
                                t_y_mask * (self.scale_to_dB(t_y_point, dB_max, dB_min) - self.scale_to_dB(t_y_point_pred, dB_max, dB_min)), axis=1) ** 2 
                                / np.sum(t_y_mask, axis=1)).tolist()
                    losses += loss_
                    print(f'{np.sqrt(np.mean(loss_))}')
                    
            return torch.sqrt(torch.Tensor(losses).mean())
        

    def scale_to_dB(self, value, dB_max, dB_min):
        range_dB = dB_max - dB_min
        dB = value * range_dB + dB_min
        return dB
    

    def save_model(self, out_path):
        torch.save(self, out_path)