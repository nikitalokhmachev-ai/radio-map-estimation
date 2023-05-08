import torch.nn as nn
import torch.nn.functional as F

class DiffConv2d(nn.Module):
    def __init__(self, enc_in, n_dim, kernel_size, padding, temb_channels, nonlinearity):
        super().__init__()
        self.conv2d = nn.Conv2d(enc_in, n_dim, kernel_size=kernel_size, padding=padding)
        self.time_emb_proj = nn.Linear(temb_channels, n_dim)
        if nonlinearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif nonlinearity == "mish":
            self.nonlinearity = nn.Mish()
        elif nonlinearity == "silu":
            self.nonlinearity = nn.SiLU()
        elif nonlinearity == "gelu":
            self.nonlinearity = nn.GELU()
        elif nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()

    def forward(self, x, emb):
        hidden_states = self.conv2d(x)
        temb = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None] 
        hidden_states = hidden_states + temb
        return self.nonlinearity(hidden_states)
    

