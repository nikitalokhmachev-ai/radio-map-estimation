from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class UNet_V3_Transmitter(nn.Module):

    def __init__(self, in_channels=2, latent_channels=64, out_channels=1, features=[32,32,64]):
        # TODO: Figure out the best way to call super().__init__() here. 
        super(UNet_V3_Transmitter, self).__init__()

        if isinstance(features, int):
            features = [features] * 3

        # Use the same 3-layer blocks as UNet_V2
        self.encoder1 = UNet_V3_Transmitter._block(in_channels, features[0], name="enc1")
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet_V3_Transmitter._block(features[0], features[1], name="enc2")
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet_V3_Transmitter._block(features[1], features[2], name="enc3")
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet_V3_Transmitter._block(features[2], latent_channels, name='bottleneck')

        self.upconv3 = nn.ConvTranspose2d(latent_channels, features[2], kernel_size=2, stride=2)
        self.decoder3 = UNet_V3_Transmitter._block(features[2] * 2, features[2], name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = UNet_V3_Transmitter._block(features[1] * 2, features[1], name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = UNet_V3_Transmitter._block(features[0] * 2, features[0], name="dec1")

        # Here is one extra 1x1 Convolution to change the output into the right shape
        self.conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        map = torch.sigmoid(self.conv(dec1))
        return map

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

            # Transform tx_loc into one-hot maps, concatenate to t_x_point
            batch_ = torch.arange(0,t_y_point.shape[0]).to(torch.int)
            channels_ = torch.zeros(t_y_point.shape[0]).to(torch.int)
            x_coord = torch.round(tx_loc[:,0] * t_y_point.shape[-1]).detach().to(torch.int)
            y_coord = torch.round(-tx_loc[:,1] * t_y_point.shape[-2]).detach().to(torch.int) - 1 # -1 to account for the fact that counting from top starts at 0 but counting from bottom starts from -1 (instead of -0)
            tx_loc_map = torch.zeros_like(t_y_point).to(device)
            tx_loc_map[batch_, channels_, y_coord, x_coord] = 1

            input = torch.cat((t_x_point, tx_loc_map), dim=1).to(torch.float32)

            t_y_point_pred = self.forward(input).to(torch.float32)

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
    

    def fit_wandb(self, train_dl, test_dl, scaler, optimizer, scheduler, project_name, run_name, epochs=100, 
                  save_model_epochs=25, save_model_dir='/content', use_true_evaluation=True, dB_max=-47.84, dB_min=-147):
        import wandb
        wandb.init(project=project_name, name=run_name)
        benchmark = dict()
        count = 0
        for epoch in range(epochs):
            train_running_loss = 0.0
            for i, batch in enumerate(train_dl):
                loss = self.step(batch, optimizer, train=True)
                train_running_loss += loss.detach().item()
                train_loss = train_running_loss/(i+1)
                print(f'{loss}, [{epoch + 1}, {i + 1:5d}] loss: {train_loss}')


            if use_true_evaluation:
                test_loss = self.evaluate(test_dl, scaler, dB_max, dB_min, no_scale=False)
                print(f'{test_loss}, [{epoch + 1}]')

            else:
                test_running_loss = 0.0
                for i, batch in enumerate(test_dl):
                    loss = self.step(batch, optimizer, train=False)
                    test_running_loss += loss.detach().item()
                    test_loss = test_running_loss/(i+1)
                    print(f'{loss}, [{epoch + 1}, {i + 1:5d}] loss: {test_loss}')
                                    
            wandb.log({'train_loss': train_loss, 'test_loss': test_loss})

            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                filepath = os.path.join(save_model_dir, f'epoch_{epoch}.pth')
                self.save_model(filepath)
                benchmark[count] = test_loss
                if count > 0:
                    if test_loss > benchmark[count-1]:
                        break
                count += 1


            if scheduler:
                scheduler.step()


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