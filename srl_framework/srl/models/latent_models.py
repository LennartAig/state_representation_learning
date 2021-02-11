import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from srl_framework.srl.models.base import BaseModelSRL
from srl_framework.utils.encoder import CnnEncoder, ResNetEncoder, PixelEncoder
from srl_framework.utils.decoder import CnnDecoder, ResNetDecoder, PixelDecoder


class Encoder(BaseModelSRL):
    def __init__(
        self,
        img_channels,
        state_dim,
        encoder_args=None,
        lr=0.001,
        device=None,
        img_size=84,
        normalized_obs=True,
    ):
        super(Encoder, self).__init__()

        if encoder_args["architecture"] == "impala":
            self.encoder = ResNetEncoder(
                img_channels=img_channels,
                feature_dim=state_dim,
                img_size=img_size,
                normalized_obs=normalized_obs,
                squash_latent=encoder_args["squash_latent"],
                normalize=encoder_args["normalize"],
                conv_init=encoder_args["conv_init"],
                linear_init=encoder_args["linear_init"],
            )
        elif encoder_args["architecture"] == "standard":
            self.encoder = PixelEncoder((img_channels, img_size, img_size), state_dim)
        else:
            self.encoder = CnnEncoder(
                img_channels=img_channels,
                feature_dim=state_dim,
                img_size=img_size,
                architecture=encoder_args["architecture"],
                normalized_obs=normalized_obs,
                squash_latent=encoder_args["squash_latent"],
                normalize=encoder_args["normalize"],
                conv_init=encoder_args["conv_init"],
                linear_init=encoder_args["linear_init"],
                activation=encoder_args["activation"],
                batchnorm=encoder_args["batchnorm"],
                pool=encoder_args["pool"],
                dropout=encoder_args["dropout"],
            )

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()

    def get_state(self, obs, grad=True):
        state = self.encode(obs, grad=grad)
        return state

    def encode(self, x, grad=True):
        detach = not grad
        z = self.encoder(x)
        return z

    def forward(self, x):
        z = self.encode(x)
        return z


class AE(BaseModelSRL):
    def __init__(
        self,
        img_channels,
        state_dim,
        encoder_args=None,
        lr=0.001,
        device=None,
        decoder_dim=0,
        img_size=84,
        normalized_obs=True,
    ):
        super(AE, self).__init__()
        self.decoder_dim = state_dim if decoder_dim == 0 else decoder_dim

        if encoder_args["architecture"] == "impala":
            self.encoder = ResNetEncoder(
                img_channels=img_channels,
                feature_dim=state_dim,
                img_size=img_size,
                normalized_obs=normalized_obs,
                squash_latent=encoder_args["squash_latent"],
                normalize=encoder_args["normalize"],
                conv_init=encoder_args["conv_init"],
                linear_init=encoder_args["linear_init"],
            )

            self.decoder = ResNetDecoder(
                feature_dim=self.decoder_dim,
                in_dim=self.encoder.output_dim,
                out_dim=img_channels,
                img_size=img_size,
            )
        elif encoder_args["architecture"] == "standard":
            self.encoder = PixelEncoder((img_channels, img_size, img_size), state_dim)
            self.decoder = PixelDecoder((img_channels, img_size, img_size), state_dim)
        else:
            self.encoder = CnnEncoder(
                img_channels=img_channels,
                feature_dim=state_dim,
                img_size=img_size,
                architecture=encoder_args["architecture"],
                normalized_obs=normalized_obs,
                squash_latent=encoder_args["squash_latent"],
                normalize=encoder_args["normalize"],
                conv_init=encoder_args["conv_init"],
                linear_init=encoder_args["linear_init"],
                activation=encoder_args["activation"],
                batchnorm=encoder_args["batchnorm"],
                pool=encoder_args["pool"],
                dropout=encoder_args["dropout"],
            )

            self.decoder = CnnDecoder(
                feature_dim=self.decoder_dim,
                in_dim=self.encoder.output_dim,
                out_dim=img_channels,
                architecture=encoder_args["architecture"],
                conv_init=encoder_args["conv_init"],
                linear_init=encoder_args["linear_init"],
                activation=encoder_args["activation"],
                batchnorm=encoder_args["batchnorm"],
                pool=encoder_args["pool"],
                dropout=encoder_args["dropout"],
            )

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()

    def get_state(self, obs, detach=False):
        state = self.encode(obs, detach=detach)
        return state

    def encode(self, x, detach=False):
        z = self.encoder(x, detach=detach)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        z = self.encode(x)
        decoded = self.decode(z)
        return decoded, z


class VAE(BaseModelSRL):
    def __init__(
        self,
        img_channels,
        state_dim,
        encoder_args=None,
        lr=0.001,
        beta=1.0,
        device=None,
        decoder_dim=0,
        img_size=84,
        normalized_obs=True,
    ):
        super(VAE, self).__init__()
        self.decoder_dim = state_dim if decoder_dim == 0 else decoder_dim
        self.beta = beta

        if encoder_args["architecture"] == "impala":
            self.encoder = ResNetEncoder(
                img_channels=img_channels,
                feature_dim=2 * state_dim,
                img_size=img_size,
                normalized_obs=normalized_obs,
                squash_latent=encoder_args["squash_latent"],
                normalize=encoder_args["normalize"],
                conv_init=encoder_args["conv_init"],
                linear_init=encoder_args["linear_init"],
            )

            self.decoder = ResNetDecoder(
                feature_dim=self.decoder_dim,
                in_dim=self.encoder.output_dim,
                out_dim=img_channels,
                img_size=img_size,
            )
        elif encoder_args["architecture"] == "standard":
            self.encoder = PixelEncoder(
                (img_channels, img_size, img_size), state_dim, vae=True
            )
            print(int(state_dim / 2))
            self.decoder = PixelDecoder((img_channels, img_size, img_size), state_dim)
        else:
            self.encoder = CnnEncoder(
                img_channels=img_channels,
                feature_dim=2 * state_dim,
                img_size=img_size,
                architecture=encoder_args["architecture"],
                normalized_obs=normalized_obs,
                squash_latent=encoder_args["squash_latent"],
                normalize=encoder_args["normalize"],
                conv_init=encoder_args["conv_init"],
                linear_init=encoder_args["linear_init"],
                activation=encoder_args["activation"],
                batchnorm=encoder_args["batchnorm"],
                pool=encoder_args["pool"],
                dropout=encoder_args["dropout"],
            )

            self.decoder = CnnDecoder(
                feature_dim=self.decoder_dim,
                in_dim=self.encoder.output_dim,
                out_dim=img_channels,
                architecture=encoder_args["architecture"],
                conv_init=encoder_args["conv_init"],
                linear_init=encoder_args["linear_init"],
                activation=encoder_args["activation"],
                batchnorm=encoder_args["batchnorm"],
                pool=encoder_args["pool"],
                dropout=encoder_args["dropout"],
            )

        self.h_dim = state_dim
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()

    def get_state(self, obs, detach=False, deterministic=True):
        z, _, _ = self.encode(obs, detach=detach)
        return z

    def encode(self, obs, detach=False):
        z, mu, logvar = self.encoder.forward_latent(obs, detach=detach)
        return z, mu, logvar

    def decode(self, z):
        """
        """
        # hidden_var = self.decoder_linear(z)
        x = self.decoder(z)
        return x

    def reparameterize(self, mu, logvar):
        """
        Reparameterize for the backpropagation of z instead of q.
        (See "The reparameterization trick" section of https://arxiv.org/abs/1312.6114)
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        """
        """
        z, mu, logvar = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar, z


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        state_size: int = 50,
        beta: float = 0.25,
    ):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)
        # (H,W): 64 -> (16,16) 84 -> (21,21)

    def forward(self, latents):
        latents = latents.permute(
            0, 2, 3, 1
        ).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape

        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = (
            torch.sum(flat_latents ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        )  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]
        state = encoding_inds.view(latents_shape[0], -1)

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(
            encoding_one_hot, self.embedding.weight
        )  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return (
            quantized_latents.permute(0, 3, 1, 2).contiguous(),
            vq_loss,
            state,
        )  # [B x D x H x W]


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, input):
        return input + self.resblock(input)


class VQ_VAE(BaseModelSRL):
    """
    Refernce:
    [2] https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
    """

    def __init__(
        self,
        in_channels: int = 3,
        state_size: int = 50,
        embedding_dim: int = 64,
        num_embeddings: int = 256,
        params=None,
        device=None,
        hidden_dims: list = None,
        beta: float = 0.25,
        img_size: int = 64,
        **kwargs
    ) -> None:
        super(VQ_VAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta
        self.in_channels = in_channels
        # self.output_layer = nn.Linear(h_dim, feature_dim)

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU(),
            )
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1
                ),
                nn.LeakyReLU(),
            )
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    out_channels=self.in_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.Tanh(),
            )
        )

        self.decoder = nn.Sequential(*modules)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)  # TODO
        self.train()

    def get_state(self, obs, grad=True, deterministic=False):
        if grad:
            encoding = self.encode(obs)[0]
            quantized_inputs, vq_loss, state = self.vq_layer(encoding)
        else:
            with torch.no_grad():
                encoding = self.encode(obs)[0]
            quantized_inputs, vq_loss, state = self.vq_layer(encoding)
        state = state.float() / 255.0
        return state

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input, **kwargs):
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss, state = self.vq_layer(encoding)
        return self.decode(quantized_inputs), vq_loss, state

    def loss_function(self, *args, **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "VQ_Loss": vq_loss}

    def sample(self, num_samples: int, current_device: [int, str], **kwargs):
        raise Warning("VQVAE sampler is not implemented.")

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class VAE_2(BaseModelSRL):
    """
    """

    def __init__(self, img_channels, state_dim, param, device=None):
        super(VAE_2, self).__init__()
        self.encoder = CnnEncoder(
            img_channels=img_channels, feature_dim=2 * state_dim, params=param.CNN
        )
        self.decoder = CnnDecoder(
            feature_dim=state_dim,
            in_dim=self.encoder.output_dim,
            out_dim=img_channels,
            params=param.CNN,
        )
        self.layer_norm = nn.LayerNorm(state_dim)

        # Set Model to device
        self.to(self.device)
        self.state_size = state_dim

        self.optimizer = optim.Adam(self.parameters(), lr=param.LEARNING_RATE)

        self.normalized_latent = param.NORMALIZED_LATENT
        self.squashed_latent = param.SQUASHED_LATENT
        self.train

    def get_state(self, obs, grad=True, deterministic=False):
        """
        Input:
        ------
            - obs (torch tensor)
            - grad (bool): Set if gradient is required in latter calculations
        Return:
        ------
            - torch tensor
        # TODO: mu + std
        """
        if grad:
            if deterministic:
                state = self.encode(obs)[0]
            else:
                mu, logvar = self.encode(obs)
                state = self.reparameterize(mu, logvar)
        else:
            with torch.no_grad():
                if deterministic:
                    state = self.encode(obs)[0]
                else:
                    mu, logvar = self.encode(obs)
                    state = self.reparameterize(mu, logvar)

        # if self.normalized_latent: state = self.layer_norm(state)
        # if self.squashed_latent: state = torch.tanh(state)
        return state

    def encode(self, x):
        """
        """
        hidden_var = self.encoder(x)
        mu = hidden_var[:, self.state_size]
        logvar = self.logvar_net(hidden_var)
        if self.normalized_latent:
            mu = self.layer_norm(mu)
            logvar = self.layer_norm(logvar)
        if self.squashed_latent:
            mu = torch.tanh(mu)
            logvar = torch.tanh(logvar)
        return mu, logvar

    def decode(self, z):
        """
        """
        hidden_var = self.decoder_linear(z)
        x = self.decoder(hidden_var)
        return x

    def reparameterize(self, mu, logvar):
        """
        Reparameterize for the backpropagation of z instead of q.
        (See "The reparameterization trick" section of https://arxiv.org/abs/1312.6114)
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        """
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar
