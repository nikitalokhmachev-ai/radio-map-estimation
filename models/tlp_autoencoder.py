import torch
import numpy as np
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from .autoencoder import Autoencoder

class TLPAutoencoder(Autoencoder):
    def __init__(self):
        super().__init__()

        self.encoder = None
        self.decoder = None

    def forward(self, x):
        pass
        '''
        Forward pass will differ based on whether encoder or decoder calculates tx_loc.

        # Encoder
        x, tx_loc, skip1, skip2, skip3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3)
        return x, tx_loc

        # Decoder
        x, skip1, skip2, skip3 = self.encoder(x)
        x, tx_loc = self.decoder(x, skip1, skip2, skip3)
        '''
        
    
    def step(self, batch, optimizer, w_rec=0.5, w_loc=0.5, train=True):
        with torch.set_grad_enabled(train):
            t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, tx_loc = batch
            t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.to(torch.float32).to(device), t_y_mask.to(torch.float32).to(device)
            t_channel_pow = t_channel_pow.to(torch.float32).to(device)
            tx_loc = tx_loc.to(torch.float32).to(device)

            t_y_point_pred, tx_loc_pred = self.forward(t_x_point)
            t_y_point_pred = t_y_point_pred.to(torch.float32)
            tx_loc_pred = tx_loc_pred.to(torch.float32)

            rec_loss_ = torch.nn.functional.mse_loss(t_y_point * t_y_mask, t_y_point_pred * t_y_mask).to(torch.float32)

            if self.loc_loss_func == 'mse':
                loss_func = torch.nn.MSELoss()
                loc_loss_ = loss_func(tx_loc, tx_loc_pred).to(torch.float32)
            
            else:
                # Transform tx_loc into one-hot maps
                batch_ = torch.arange(0,tx_loc_pred.shape[0]).to(torch.int)
                channels_ = torch.zeros(tx_loc_pred.shape[0]).to(torch.int)
                x_coord = torch.round(tx_loc[:,0] * tx_loc_pred.shape[-1]).detach().to(torch.int)
                y_coord = torch.round(-tx_loc[:,1] * tx_loc_pred.shape[-2]).detach().to(torch.int)
                tx_loc_map = torch.zeros_like(tx_loc_pred)
                tx_loc_map[batch_, channels_, y_coord, x_coord] = 1
                tx_loc_map = tx_loc_map.to(device)

                if self.loc_loss_func == 'bce':
                    # Weight BCE Loss by number of negative pixels to positive pixels
                    pixels = t_x_point.shape[-2] * t_x_point.shape[-1]
                    pos_weight = torch.Tensor([pixels]).to(device)
                    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
                    loc_loss_ = loss_func(tx_loc_map, tx_loc_pred)

                if self.loc_loss_func == 'softmax':
                    # Flatten tx_loc_map and tx_loc_pred
                    tx_loc_map = tx_loc_map.flatten(1)
                    tx_loc_pred = tx_loc_pred.flatten(1)
                    loss_func = torch.nn.CrossEntropyLoss()
                    loc_loss_ = loss_func(tx_loc_map, tx_loc_pred)

            loss_ = w_rec * rec_loss_ + w_loc * loc_loss_
            
            if train:
                loss_.backward()
                optimizer.step()
                optimizer.zero_grad()

        return loss_, rec_loss_, loc_loss_
            
    
    def fit(self, train_dl, optimizer, w_rec=0.5, w_loc=0.5, epochs=100, save_model_epochs=25, save_model_dir ='/content'):
        for epoch in range(epochs):
            running_loss = 0.0
            rec_running_loss = 0.0
            loc_running_loss = 0.0
            for i, batch in enumerate(train_dl):
                loss_, rec_loss_, loc_loss = self.step(batch, optimizer, w_rec, w_loc, train=True)
                running_loss += loss_.detach().item()
                rec_running_loss += rec_loss_.detach().item()
                loc_running_loss += loc_loss.detach().item()
                print(f'{loss_}, [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1)}, reconstruction_loss: {rec_running_loss/(i+1)}, location_loss: {loc_running_loss/(i+1)}')

            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                filepath = os.path.join(save_model_dir, f'epoch_{epoch}.pth')
                self.save_model(filepath)

        return running_loss / (i+1)
    
    def fit_wandb(self, train_dl, test_dl, optimizer, project_name, run_name, w_rec=0.5, w_loc=0.5, epochs=100, save_model_epochs=25, save_model_dir='/content'):
        import wandb
        wandb.init(project=project_name, name=run_name)
        for epoch in range(epochs):
            running_loss, rec_running_loss, loc_running_loss = 0.0, 0.0, 0.0
            for i, batch in enumerate(train_dl):
                loss, rec_loss, loc_loss = self.step(batch, optimizer, w_rec, w_loc, train=True)
                running_loss += loss.detach().item()
                rec_running_loss += rec_loss.detach().item()
                loc_running_loss += loc_loss.detach().item()
                train_loss = running_loss/(i+1)
                train_rec_loss = rec_running_loss/(i+1)
                train_loc_loss = loc_running_loss/(i+1)
                print(f'{loss}, [{epoch + 1}, {i + 1:5d}] loss: {train_loss}, reconstruction_loss: {train_rec_loss}, location_loss: {train_loc_loss}')
            
            wandb.log({'train_loss': train_loss, 'train_reconstruction_loss': train_rec_loss, 'train_location_loss':train_loc_loss})
        
            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                filepath = os.path.join(save_model_dir, f'epoch_{epoch}.pth')
                self.save_model(filepath)

            running_loss, rec_running_loss, loc_running_loss = 0.0, 0.0, 0.0
            for i, batch in enumerate(test_dl):
                loss, rec_loss, loc_loss = self.step(batch, optimizer, w_rec, w_loc, train=False)
                running_loss += loss.detach().item()
                rec_running_loss += rec_loss.detach().item()
                loc_running_loss += loc_loss.detach().item()
                test_loss = running_loss/(i+1)
                test_rec_loss = rec_running_loss/(i+1)
                test_loc_loss = loc_running_loss/(i+1)
                print(f'{loss}, [{epoch + 1}, {i + 1:5d}] loss: {test_loss}, reconstruction_loss: {test_rec_loss}, location_loss: {test_loc_loss}')
                
            wandb.log({'test_loss': test_loss, 'test_reconstruction_loss': test_rec_loss, 'test_location_loss':test_loc_loss})