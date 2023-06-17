from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import os

from .unet import UNet, UNet_V2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs,  targets, reduction='none')
        loss = (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss.mean()


# UNetFocal_V2 is named to match the version number of TLPResUNetBCE_V2. There is no V1.
class UNetFocal_V2(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()

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

        self.conv_map = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.conv_tx = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.focal_loss = FocalLoss(gamma=2)

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
        map = torch.sigmoid(self.conv_map(dec1))
        tx_loc = self.conv_tx(dec1)
        return map, tx_loc

    
    def step(self, batch, optimizer, w_rec, w_loc, train=True):
        with torch.set_grad_enabled(train):
            t_x_point, t_y_point, t_y_mask, _, _, tx_loc = batch
            t_x_point, t_y_point = t_x_point.to(torch.float32).to(device), t_y_point.to(torch.float32).to(device)
            t_y_mask, tx_loc = t_y_mask.to(torch.float32).to(device), tx_loc.to(torch.float32).to(device)

            t_y_point_pred, tx_loc_pred = self.forward(t_x_point)
            t_y_point_pred = t_y_point_pred.to(torch.float32)
            tx_loc_pred = tx_loc_pred.to(torch.float32)

            # Transform tx_loc into one-hot maps
            batch_ = torch.arange(0,tx_loc_pred.shape[0]).to(torch.int)
            channels_ = torch.zeros(tx_loc_pred.shape[0]).to(torch.int)
            x_coord = torch.round(tx_loc[:,0] * tx_loc_pred.shape[-1]).detach().to(torch.int)
            y_coord = torch.round(-tx_loc[:,1] * tx_loc_pred.shape[-2]).detach().to(torch.int) - 1 # -1 to account for the fact that counting from top starts at 0 but counting from bottom starts from -1 (instead of -0)
            tx_loc_map = torch.zeros_like(tx_loc_pred).to(device)
            tx_loc_map[batch_, channels_, y_coord, x_coord] = 1

            rec_loss_ = nn.functional.mse_loss(t_y_point_pred * t_y_mask, t_y_point * t_y_mask).to(torch.float32)
            loc_loss_ = self.focal_loss(tx_loc_pred, tx_loc_map).to(torch.float32)

            loss_ = w_rec * rec_loss_ + w_loc * loc_loss_
            
            if train:
                loss_.backward()
                optimizer.step()
                optimizer.zero_grad()

        return loss_, rec_loss_, loc_loss_
    
    def fit(self, train_dl, optimizer, w_rec, w_loc, epochs=100, save_model_epochs=25, save_model_dir ='/content'):
        for epoch in range(epochs):
            running_loss, rec_running_loss, loc_running_loss = 0.0, 0.0, 0.0
            for i, batch in enumerate(train_dl):
                loss_, rec_loss_, loc_loss_ = self.step(batch, optimizer, w_rec=w_rec, w_loc=w_loc, train=True)
                running_loss += loss_.detach().item()
                rec_running_loss += rec_loss_.detach().item()
                loc_running_loss += loc_loss_.detach().item()
                print(f'{loss_}, [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1)}, reconstruction_loss: {rec_running_loss/(i+1)}, location_loss: {loc_running_loss/(i+1)}')

            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                filepath = os.path.join(save_model_dir, f'epoch_{epoch}.pth')
                self.save_model(filepath)
    

    def fit_wandb(self, train_dl, test_dl, optimizer, project_name, run_name, w_rec, w_loc, epochs=100, save_model_epochs=25, save_model_dir='/content'):
        import wandb
        wandb.init(project=project_name, name=run_name)
        for epoch in range(epochs):
            train_running_loss, train_rec_running_loss, train_loc_running_loss = 0.0, 0.0, 0.0
            for i, batch in enumerate(train_dl):
                loss_, rec_loss_, loc_loss_ = self.step(batch, optimizer, w_rec=w_rec, w_loc=w_loc, train=True)
                train_running_loss += loss_.detach().item()
                train_rec_running_loss += rec_loss_.detach().item()
                train_loc_running_loss += loc_loss_.detach().item()
                train_loss = train_running_loss / (i+1)
                train_rec_loss = train_rec_running_loss / (i+1)
                train_loc_loss = train_loc_running_loss / (i+1)
                print(f'{loss_}, [{epoch + 1}, {i + 1:5d}] loss: {train_loss}, reconstruction_loss: {train_rec_loss}, location_loss: {train_loc_loss}')
                    
            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                filepath = os.path.join(save_model_dir, f'epoch_{epoch}.pth')
                self.save_model(filepath)

            test_running_loss, test_rec_running_loss, test_loc_running_loss = 0.0, 0.0, 0.0
            for i, batch in enumerate(test_dl):
                loss_, rec_loss_, loc_loss_ = self.step(batch, optimizer, w_rec=w_rec, w_loc=w_loc, train=False)
                test_running_loss += loss_.detach().item()
                test_rec_running_loss += rec_loss_.detach().item()
                test_loc_running_loss += loc_loss_.detach().item()
                test_loss = test_running_loss / (i+1)
                test_rec_loss = test_rec_running_loss / (i+1)
                test_loc_loss = test_loc_running_loss / (i+1)
                print(f'{loss_}, [{epoch + 1}, {i + 1:5d}] loss: {test_loss}, reconstruction_loss: {test_rec_loss}, location_loss: {test_loc_loss}')
                
            wandb.log({'train_loss': train_rec_loss, 'train_location_loss': train_loc_loss, 'train_combined_loss': train_loss,
                       'test_loss': test_rec_loss, 'test_location_loss': test_loc_loss, 'test_combined_loss': test_loss})


    def evaluate(self, test_dl, scaler, dB_max=-47.84, dB_min=-147, no_scale=False):
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_dl):
                    t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                    t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                    t_channel_pow = t_channel_pow.flatten(1).to(device).detach().cpu().numpy()
                    t_y_point_pred, _ = self.forward(t_x_point)
                    t_y_point_pred = t_y_point_pred.flatten(1).detach().cpu().numpy()
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



# This model splits off the Map Reconstruction and Transmitter Localization Heads at the Latent Space / Beginning of Decoder
class UNetFocal_V3(UNetFocal_V2):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNetFocal_V2, self).__init__()

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

        # Signal Map Deconvolution
        self.upconv4_map = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4_map = UNet._block((features * 8) * 2, features * 8, name="dec4_map")
        self.upconv3_map = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3_map = UNet._block((features * 4) * 2, features * 4, name="dec3_map")
        self.upconv2_map = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2_map = UNet._block((features * 2) * 2, features * 2, name="dec2_map")
        self.upconv1_map = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1_map = UNet._block(features * 2, features, name="dec1_map")
        self.conv_map = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)


        # Transmitter Location Deconvolution
        self.upconv4_tx = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4_tx = UNet._block((features * 8) * 2, features * 8, name="dec4_tx")
        self.upconv3_tx = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3_tx = UNet._block((features * 4) * 2, features * 4, name="dec3_tx")
        self.upconv2_tx = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2_tx = UNet._block((features * 2) * 2, features * 2, name="dec2_tx")
        self.upconv1_tx = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1_tx = UNet._block(features * 2, features, name="dec1_tx")
        self.conv_tx = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

        self.focal_loss = FocalLoss(gamma=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4_map = self.upconv4_map(bottleneck)
        dec4_map = torch.cat((dec4_map, enc4), dim=1)
        dec4_map = self.decoder4_map(dec4_map)
        dec3_map = self.upconv3_map(dec4_map)
        dec3_map = torch.cat((dec3_map, enc3), dim=1)
        dec3_map = self.decoder3_map(dec3_map)
        dec2_map = self.upconv2_map(dec3_map)
        dec2_map = torch.cat((dec2_map, enc2), dim=1)
        dec2_map = self.decoder2_map(dec2_map)
        dec1_map = self.upconv1_map(dec2_map)
        dec1_map = torch.cat((dec1_map, enc1), dim=1)
        dec1_map = self.decoder1_map(dec1_map)
        map = torch.sigmoid(self.conv_map(dec1_map))

        dec4_tx = self.upconv4_tx(bottleneck)
        dec4_tx = torch.cat((dec4_tx, enc4), dim=1)
        dec4_tx = self.decoder4_tx(dec4_tx)
        dec3_tx = self.upconv3_tx(dec4_tx)
        dec3_tx = torch.cat((dec3_tx, enc3), dim=1)
        dec3_tx = self.decoder3_tx(dec3_tx)
        dec2_tx = self.upconv2_tx(dec3_tx)
        dec2_tx = torch.cat((dec2_tx, enc2), dim=1)
        dec2_tx = self.decoder2_tx(dec2_tx)
        dec1_tx = self.upconv1_tx(dec2_tx)
        dec1_tx = torch.cat((dec1_tx, enc1), dim=1)
        dec1_tx = self.decoder1_tx(dec1_tx)
        tx_loc = self.conv_tx(dec1_tx)

        return map, tx_loc


class UNet_V3_Focal_V2(UNetFocal_V2):
    '''UNetFocal_V2 reimplemented for UNet_V3 architecture'''

    def __init__(self, in_channels=2, latent_channels=64, out_channels=1, features=[32,32,64]):
        super(UNetFocal_V2, self).__init__()

        if isinstance(features, int):
            features = [features] * 3

        # Use the same 3-layer blocks as UNet_V2
        self.encoder1 = UNet_V2._block(in_channels, features[0], name="enc1")
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet_V2._block(features[0], features[1], name="enc2")
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet_V2._block(features[1], features[2], name="enc3")
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet_V2._block(features[2], latent_channels, name='bottleneck')

        self.upconv3 = nn.ConvTranspose2d(latent_channels, features[2], kernel_size=2, stride=2)
        self.decoder3 = UNet_V2._block(features[2] * 2, features[2], name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = UNet_V2._block(features[1] * 2, features[1], name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = UNet_V2._block(features[0] * 2, features[0], name="dec1")

        # Split off Map and TX_Loc heads at final convolution
        self.conv_map = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)
        self.conv_tx = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

        # Focal Loss as a weighted form of BCE Loss
        self.focal_loss = FocalLoss(gamma=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        map = torch.sigmoid(self.conv_map(dec1))
        tx_loc = self.conv_tx(dec1)
        return map, tx_loc