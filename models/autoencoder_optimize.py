import os
import torch
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Autoencoder_Optimize(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = None#Encoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = None#Decoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
    def fit(self, train_dl, optimizer, epochs=100, loss='mse'):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_dl):
                optimizer.zero_grad()
                t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(torch.float32).to(device), t_y_mask.flatten(1).to(torch.float32).to(device)
                t_channel_pow = t_channel_pow.flatten(1).to(torch.float32).to(device)
                t_y_point_pred = self.forward(t_x_point).to(torch.float32)
                loss_ = torch.nn.functional.mse_loss(t_y_point * t_y_mask, t_y_point_pred * t_y_mask).to(torch.float32)
                if loss == 'rmse':
                    loss_ = torch.sqrt(loss_)
                loss_.backward()
                optimizer.step()

                running_loss += loss_.item()        
                print(f'{loss_}, [{epoch + 1}, {i + 1:5d}] loss: {running_loss/(i+1)}')

        return running_loss / (i+1)


    def evaluate(self, test_dl, scaler, dB_max=-47.84, dB_min=-147, no_scale=False):
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_dl):
                    t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                    t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                    t_channel_pow = t_channel_pow.flatten(1).to(device).detach().cpu().numpy()
                    t_y_point_pred = self.forward(t_x_point).detach().cpu().numpy()
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

            if scaler is None and no_scale==True:
                return torch.Tensor(losses).mean()        
            return torch.sqrt(torch.Tensor(losses).mean())
        

    def scale_to_dB(self, value, dB_max, dB_min):
        range_dB = dB_max - dB_min
        dB = value * range_dB + dB_min
        return dB
    
    def wandb_alart(self, project_name, run_name):
        import wandb
        wandb.init(project_name=project_name, name=run_name)
        wandb.alert(
            title="Test Alert", 
            text="Is this working?"
            )

    def fit_wandb(self, train_dl, test_dl, scaler, optimizer, project_name, run_name, epochs=100, save_model_epochs=25, save_model_dir='/content', loss='mse'):
        import wandb
        wandb.init(project=project_name, name=run_name)

        for epoch in range(epochs):
            train_loss = self.fit(train_dl, optimizer, epochs=1, loss=loss)
            test_loss = self.evaluate(test_dl, scaler, no_scale=True)
            wandb.log({'train_loss': train_loss, 'test_loss': test_loss})
            if (epoch + 1) % save_model_epochs == 0 or epoch == epochs - 1:
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                filepath = os.path.join(save_model_dir, f'epoch_{epoch}.pth')
                self.save_model(filepath)

    def save_model(self, out_path):
        torch.save(self, out_path)