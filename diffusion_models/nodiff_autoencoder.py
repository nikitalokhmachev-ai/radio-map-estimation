import torch
import os
from diffusers import UNet2DModel
import matplotlib.pyplot as plt
import numpy as np 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from typing import Optional, Tuple, Union
from diffusers import UNet2DModel
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from dataclasses import dataclass

@dataclass
class UNet2DOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    sample: torch.FloatTensor

class UNet(torch.nn.Module):
    def __init__(
        self,
        noise = None,
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
        add_attention = True, 
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

      self.noise = noise

      timestep_input_dim = block_out_channels[0]
      self.model.time_proj = Timesteps(timestep_input_dim, flip_sin_to_cos, freq_shift)
      self.model.time_embedding = TimestepEmbedding(timestep_input_dim, timestep_input_dim*2).to(device)
      
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): (batch) timesteps
            class_labels (`torch.FloatTensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a plain tuple.
        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`: [`~models.unet_2d.UNet2DOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        # 0. center input if necessary
        if self.model.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.model.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.model.dtype)
        emb = self.model.time_embedding(t_emb)

        # 2. pre-process
        skip_sample = sample
        sample = self.model.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.model.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.model.mid_block(sample, emb)
        # 5. up
        skip_sample = None
        for upsample_block in self.model.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        
        # 6. post-process
        sample = self.model.conv_norm_out(sample)
        sample = self.model.conv_act(sample)
        sample = self.model.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.model.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps
        
        if not return_dict:
            return (sample)

        return UNet2DOutput(sample=sample)


    def step(self, batch, optimizer=None, lr_scheduler=None, train=True):
        with torch.set_grad_enabled(train):
            t_x_points, _, _, t_channel_pows, _, i = batch
            clean_images = t_channel_pows.to(torch.float32).to(device) * 2 - 1
            sample_maps = t_x_points[:,0].to(torch.float32).to(device).unsqueeze(1) * 2 - 1
            environment_masks = t_x_points[:,1].to(torch.float32).to(device).unsqueeze(1)

            # Sample noise to add to the images
            if self.noise:
                noise = self.noise
            else:
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.zeros(bs).long().to(device)

            model_input = torch.cat((clean_images, sample_maps, environment_masks), 1)
            
            # Predict the noise residual
            noise_pred = self.model(model_input, timesteps, return_dict=False)[0][:,0:1]
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            if train:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        return loss

    def fit(self, config, optimizer, train_dataloader, lr_scheduler):
        # Now you train the model
        for epoch in range(config.num_epochs):
            running_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                loss = self.step(batch, optimizer, lr_scheduler)
                running_loss += loss.detach().item()
                print(f'{loss}, [{epoch + 1}, {step + 1:5d}] loss: {running_loss/(step+1)}')
            
            # After each epoch you optionally sample some demo images with evaluate() and save the model
        
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                self.save_model(config, f'epoch_{epoch}.pth')

        return running_loss / (step+1)

    def fit_wandb(self, project_name, run_name, config, optimizer, train_dataloader, lr_scheduler, test_dataloader):
            import wandb
            wandb.init(project=project_name, name=run_name)

            for epoch in range(config.num_epochs):
                running_loss = 0.0
                for step, batch in enumerate(train_dataloader):
                    loss = self.step(batch, optimizer, lr_scheduler, train=True)
                    running_loss += loss.detach().item()
                    train_loss = running_loss / (step+1)
                    print(f'{loss}, [{epoch + 1}, {step + 1:5d}] train loss: {train_loss}')
                
                wandb.log({'train_loss': train_loss})
            
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    self.save_model(config, f'epoch_{epoch}.pth')

                running_loss = 0.0
                for step, batch in enumerate(test_dataloader):
                    loss = self.step(batch, optimizer, lr_scheduler, train=False)
                    running_loss += loss.detach().item()
                    test_loss = running_loss / (step+1)
                    print(f'{loss}, [{step + 1:5d}] test loss: {test_loss}')
                
                wandb.log({'test_loss': test_loss})


    def evaluate(self, test_dl, scaler):
        losses = []
        with torch.no_grad():
            for i, data in enumerate(test_dl):
                    t_x_point, t_y_point, t_y_mask, t_channel_pow, file_path, j = data
                    t_x_point, t_y_point, t_y_mask = t_x_point.to(torch.float32).to(device), t_y_point.flatten(1).to(device), t_y_mask.flatten(1).to(device)
                    t_channel_pow = t_channel_pow.to(torch.float32).to(device)

                    mask = (t_x_point[:,1] == -1).unsqueeze(1).to(torch.float32)
                    x = torch.cat([t_channel_pow, mask], 1)

                    t_y_point_pred = self.forward(x).detach().cpu().numpy()
                    t_channel_pow = t_channel_pow.flatten(1).detach().cpu().numpy()
                    building_mask = (t_x_point[:,1,:,:].flatten(1) == -1).to(torch.float64).detach().cpu().numpy()
                    loss = (np.linalg.norm((1 - building_mask) * (scaler.reverse_transform(t_channel_pow) - scaler.reverse_transform(t_y_point_pred)), axis=1) ** 2 / np.sum(building_mask == 0, axis=1)).tolist()
                    losses += loss
            
                    print(f'{np.sqrt(np.mean(loss))}')
                    
            return torch.sqrt(torch.Tensor(losses).mean())
    

    def save_model(self, config, name):
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        filepath = os.path.join(config.output_dir, name)
        torch.save(self, filepath)