import torch
import os
from diffusers import UNet2DModel
import matplotlib.pyplot as plt
import numpy as np 
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
                self.plot_samples(config, epoch, noise_scheduler, data = list(map(lambda x: x[0:4], batch)), num_samples=3, fig_size=(15,5))
        
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                self.save_model(config, f'epoch_{epoch}.pth')

        return running_loss / (step+1)


    def evaluate(self, test_dl, scaler, noise_scheduler, fixed_noise=False):
        losses = []
        if fixed_noise:
            noise = torch.randn(t_channel_pows.shape).to(device)
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
    

    def plot_samples(self, config, epoch, noise_scheduler, data, num_samples=3, fig_size=(15,5)):
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        
        rows = data[0].shape[0]
        cols = 3 + num_samples
        grid = np.empty((rows, cols, config.image_size, config.image_size))

        t_x_points, t_y_points, t_y_masks, t_channel_pows, filename, index = data
        complete_maps = t_channel_pows.to(torch.float32).to(device) * 2 - 1
        sample_maps = t_x_points[:,0].to(torch.float32).to(device).unsqueeze(1) * 2 - 1
        environment_masks = t_x_points[:,1].to(torch.float32).to(device).unsqueeze(1)
        
        for i in range(rows):
            grid[i,0] = sample_maps[i,0].detach().cpu().numpy()
            grid[i,1] = environment_masks[i,0].detach().cpu().numpy()
            grid[i,2] = complete_maps[i,0].detach().cpu().numpy()

        for n in range(num_samples):
            noise = torch.randn(complete_maps.shape).to(device)
            input = torch.cat((noise, sample_maps, environment_masks), 1)
            for t in noise_scheduler.timesteps:
                with torch.no_grad():
                    noisy_residual = self.model(input, t).sample
                    previous_noisy_sample = noise_scheduler.step(noisy_residual, t, input[:,0].unsqueeze(1)).prev_sample
                    input = torch.cat((previous_noisy_sample, sample_maps, environment_masks), 1)
        images = input.clamp(-1,1).detach().cpu().numpy()

        for i in range(rows):
            grid[i,n+3] = images[i,0]

        # Make a grid out of the images
        col_titles = ['Sample Map', 'Environment Mask', 'Complete Map', 'Generated Maps']

        fig, axs = plt.subplots(rows, cols, figsize=fig_size)

        for c in range(4):
            axs[0, c].set_title(col_titles[c])

            for i in range(rows):
                for j in range(cols):
                    if j == 1:
                        axs[i,j].matshow(grid[i,j], cmap='binary', vmin=-1, vmax=1)
                        axs[i,j].axis('off')
                    else:
                        axs[i,j].matshow(grid[i,j], cmap='hot', vmin=-1, vmax=1)
                        axs[i,j].axis('off')
        fig.tight_layout()

        # Save the images
        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        plt.savefig(f"{test_dir}/{epoch:04d}.png")
        
    def fit_wandb(self, project_name, run_name, config, noise_scheduler, optimizer, train_dataloader, lr_scheduler, test_dl, scaler, fixed_noise=False):
        import wandb
        wandb.init(project=project_name, name=run_name)
        for epoch in range(config.num_epochs):
            train_loss = self.fit(config=config, 
                                  noise_scheduler=noise_scheduler, 
                                  optimizer=optimizer, 
                                  train_dataloader=train_dataloader, 
                                  lr_scheduler=lr_scheduler)
            test_loss = self.evaluate(test_dl=test_dl, 
                                      scaler=scaler, 
                                      noise_scheduler=noise_scheduler, 
                                      fixed_noise=fixed_noise)
            wandb.log({'train_loss': train_loss, 'test_loss': test_loss})

    def save_model(self, config, name):
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        filepath = os.path.join(config.output_dir, name)
        torch.save(self, filepath)