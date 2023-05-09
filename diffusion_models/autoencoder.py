import torch
import os
from diffusers import UNet2DModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DiffusionUNet(torch.nn.Module):
    def __init__(
        self,
        sample_size = 32,
        in_channels = 3,
        out_channels = 1,
        center_input_sample = False,
        time_embedding_type = "positional",
        freq_shift = 0,
        flip_sin_to_cos = True,
        down_block_types = ("DownBlock2D",
                            "DownBlock2D",
                            "DownBlock2D",
                            "DownBlock2D",
                            "AttnDownBlock2D",
                            "DownBlock2D"),
        up_block_types = ("UpBlock2D",
                          "AttnUpBlock2D",
                          "UpBlock2D",
                          "UpBlock2D",
                          "UpBlock2D",
                          "UpBlock2D"),
        block_out_channels = (16, 16, 32, 32, 64, 64),
        layers_per_block = 2,
        mid_block_scale_factor = 1,
        downsample_padding = 1,
        act_fn = "silu",
        attention_head_dim = 8,
        norm_num_groups = 16,
        norm_eps = 1e-5,
        resnet_time_scale_shift = "default",
        add_attention = True
    ):

      super().__init__()

      self.model = UNet2DModel(
          sample_size = sample_size,
          in_channels = in_channels,
          out_channels = out_channels,
          center_input_sample = center_input_sample,
          time_embedding_type = time_embedding_type,
          freq_shift = freq_shift,
          flip_sin_to_cos = flip_sin_to_cos,
          down_block_types = down_block_types,
          up_block_types = up_block_types,
          block_out_channels = block_out_channels,
          layers_per_block = layers_per_block,
          mid_block_scale_factor = mid_block_scale_factor,
          downsample_padding = downsample_padding,
          act_fn = act_fn,
          attention_head_dim = attention_head_dim,
          norm_num_groups = norm_num_groups,
          norm_eps = norm_eps,
          resnet_time_scale_shift = resnet_time_scale_shift,
          add_attention = add_attention
      ).to(device)

    def forward(self, x, t, return_dict=False):
        return self.model(x, t, return_dict=return_dict)

    def fit(self, config, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
      
        # Now you train the model
        for epoch in range(config.num_epochs):
            running_loss = 0.0
            for step, batch in enumerate(train_dataloader):
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
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.detach().item()
                print(f'{loss}, [{epoch + 1}, {step + 1:5d}] loss: {running_loss/(step+1)}')
            

            # After each epoch you optionally sample some demo images with evaluate() and save the model

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                self.plot_samples(config, epoch, model=self.model, scheduler=noise_scheduler, data=batch[0:4])
        
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                self.save_model(config.output_dir)

        return running_loss / (step+1)


    def evaluate(self, test_dl, scaler):
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_dl):
                    t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                    t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                    t_channel_pow = t_channel_pow.flatten(1).to(device).detach().cpu().numpy()
                    t_y_point_pred = self.forward(t_x_point).detach().cpu().numpy()
                    building_mask = (t_x_point[:,1,:,:].flatten(1) == -1).to(torch.float64).detach().cpu().numpy()
                    loss = (np.linalg.norm((1 - building_mask) * (scaler.reverse_transform(t_channel_pow) - scaler.reverse_transform(t_y_point_pred)), axis=1) ** 2 / np.sum(building_mask == 0, axis=1)).tolist()
                    losses += loss
            
                    print(f'{np.sqrt(np.mean(loss))}')
                    
            return torch.sqrt(torch.Tensor(losses).mean())
        
    def fit_wandb(self, train_dl, test_dl, scaler, optimizer, project_name, run_name, epochs=100, loss='mse'):
        import wandb
        wandb.init(project=project_name, name=run_name)
        for epoch in range(epochs):
            train_loss = self.fit(train_dl, optimizer, epochs=1, loss=loss)
            test_loss = self.evaluate(test_dl, scaler)
            wandb.log({'train_loss': train_loss, 'test_loss': test_loss})

    def save_model(self, out_path):
        torch.save(self, out_path)