from .resnet import Encoder as ResnetEncoder, Decoder as ResnetDecoder
from .unet import Encoder as UnetEncoder, Decoder as UnetDecoder
from .vae import Encoder as VariationalEncoder, Decoder as VariationalDecoder
from .autoencoder import Autoencoder

class ResnetAutoencoder(Autoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = ResnetEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = ResnetDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)

class VariationalAutoencoder(Autoencoder):
    def __init__(self, enc_in=2, enc_out=4, dec_out=1, n_dim=27, leaky_relu_alpha=0.3):
        super().__init__()

        self.encoder = VariationalEncoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)
        self.decoder = VariationalDecoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu_alpha)