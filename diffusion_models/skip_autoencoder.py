import torch
import os
from diffusers import UNet2DModel
import matplotlib.pyplot as plt
import numpy as np 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from autoencoder import DiffusionUNet

class SkipDiffusionUNet(DiffusionUNet):
    
    def step(self, batch, noise_scheduler, optimizer=None, lr_scheduler=None, train=True):
        with torch.set_grad_enabled(train):
            t_x_points, _, _, t_channel_pows, _, i = batch
            clean_images = t_channel_pows.to(torch.float32).to(device) * 2 - 1
            sample_maps = t_x_points[:,0].to(torch.float32).to(device).unsqueeze(1) * 2 - 1
            environment_masks = t_x_points[:,1].to(torch.float32).to(device).unsqueeze(1)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            model_input = torch.cat((noisy_images, sample_maps, environment_masks), 1)
            
            # Predict the noise residual
            noise_pred = self.model(model_input, timesteps, return_dict=False)[0]
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            if train:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        return loss


    def evaluate(self, test_dl, scaler, noise_scheduler, fixed_noise=None):
        losses = []
        if fixed_noise:
            noise = fixed_noise.to(device)
        for step, batch in enumerate(test_dl):
            t_x_points, _, _, t_channel_pows, _, _ = batch
            sample_maps = t_x_points[:,0].to(torch.float32).to(device).unsqueeze(1) * 2 - 1
            environment_masks = t_x_points[:,1].to(torch.float32).to(device).unsqueeze(1)
            # Sample noise to add to the images
            if not fixed_noise:
                noise = torch.randn(t_channel_pows.shape).to(device)

            inputs = torch.cat((noise, sample_maps, environment_masks), 1).to(device)
            for t in noise_scheduler.timesteps:
                with torch.no_grad():
                    noisy_residual = self.model(inputs, t).sample
                    previous_noisy_sample = noise_scheduler.step(noisy_residual, t, inputs[:,0].unsqueeze(1)).prev_sample
                    inputs = torch.cat((previous_noisy_sample, sample_maps, environment_masks), 1)
            t_y_point_preds = (inputs.clamp(-1,1)[:,0].detach().cpu().unsqueeze(1).flatten(1).numpy() + 1) / 2
            building_mask = (t_x_points[:,1,:,:].flatten(1) == -1).to(torch.float64).detach().cpu().numpy()
            t_channel_pows = t_channel_pows.flatten(1).to(device).detach().cpu().numpy()
            loss = (np.linalg.norm((1 - building_mask) * (scaler.reverse_transform(t_channel_pows) - scaler.reverse_transform(t_y_point_preds)), axis=1) ** 2 / np.sum(building_mask == 0, axis=1)).tolist()
            losses += loss
            print(f'{step} {np.sqrt(np.mean(loss))}')
        final_loss = torch.sqrt(torch.Tensor(losses).mean())

        return final_loss
    