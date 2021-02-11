import torch
import torch.nn as nn
import torch.nn.functional as F
from srl_framework.utils.networks import make_decoder, inits


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv2dSame(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        padding_layer=nn.ReflectionPad2d,
    ):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            Conv2dSame(in_channels, out_channels, 3),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class CnnDecoder(nn.Module):
    """
    -
    """

    def __init__(
        self,
        feature_dim=128,
        in_dim=31,
        out_dim=3,
        architecture="std",
        activation="ReLU",
        batchnorm=False,
        pool=False,
        dropout=False,
        conv_init="delta_orthogonal",
        linear_init="orthogonal",
    ):
        super(CnnDecoder, self).__init__()
        if architecture == "std":
            channels = [32, 32, 32, 32, 32, 32] + [out_dim]
            kernel_sizes = [3, 3, 5, 5, 5, 6]
            stride_size = [1, 1, 1, 1, 1, 2]
            padding_sizes = [0, 0, 0, 0, 0, 0]
        elif architecture == "std2":
            channels = [32, 32, 32, 32, 32, 32] + [out_dim]
            kernel_sizes = [3, 3, 3, 3, 3, 3]
            stride_size = [1, 1, 1, 1, 1, 2]
            padding_sizes = [0, 0, 0, 0, 0, 1]
        elif architecture == "nature":
            channels = [64, 128, 64, 32] + [out_dim]
            kernel_sizes = [3, 3, 3, 3]
            stride_size = [1, 1, 2, 3]
            padding_sizes = [0, 0, 0, 0]
        else:
            raise NotImplementedError

        self.in_dim = in_dim
        self.filter_size = channels[0]
        self.input_layer = nn.Linear(
            feature_dim, self.filter_size * in_dim * in_dim
        ).apply(inits[linear_init])

        self.decoder = make_decoder(
            channels=channels,
            kernels=kernel_sizes,
            strides=stride_size,
            paddings=padding_sizes,
            activation=activation,
            batchnorm=batchnorm,
            pool=pool,
            dropout=dropout,
            conv_init=conv_init,
        )

        self.train()

    def forward(self, x):
        # TODO: reconstruction with relu or not ?! CHECK FINAL LAYER
        x = torch.relu(self.input_layer(x))
        deconv = x.view(-1, self.filter_size, self.in_dim, self.in_dim)
        return self.decoder(deconv)


class ResNetDecoder(nn.Module):
    def __init__(self, feature_dim=128, in_dim=21, out_dim=9, img_size=84):
        super(ResNetDecoder, self).__init__()
        self.depths = [32, 32, 32, 16]
        self.initial_conv_size = self.depths[0] * in_dim * in_dim
        self.initial_conv_shape = (self.depths[0], in_dim, in_dim)
        self.initial_linear = nn.Linear(feature_dim, self.initial_conv_size)
        if img_size == 128 or img_size == 64:
            kernels = [3, 4, 4, 4]
            output_paddings = [1, 0, 0, 0]
            paddings = [0, 1, 1, 1]
        elif img_size == 84:
            kernels = [3, 3, 4, 4]
            output_paddings = [1, 0, 0, 0]
            paddings = [0, 0, 1, 1]
        else:
            raise NotImplementedError
        self.layer1 = self._make_layer(
            self.depths[0], self.depths[1], kernels[0], paddings[0], output_paddings[0]
        )
        self.layer2 = self._make_layer(
            self.depths[1], self.depths[2], kernels[1], paddings[1], output_paddings[1]
        )
        self.layer3 = self._make_layer(
            self.depths[2], self.depths[3], kernels[2], paddings[2], output_paddings[2]
        )
        self.layer4 = self._make_layer(
            self.depths[3], out_dim, kernels[2], paddings[2], output_paddings[2]
        )
        self.flatten = Flatten()
        self.train()

    def _make_layer(
        self, in_channels, depth, kernelsize=3, padding=0, output_padding=0
    ):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                depth,
                kernelsize,
                stride=2,
                output_padding=output_padding,
                padding=padding,
            ),
            # nn.ReflectionPad2d((1,1,1,1)),
            nn.ReLU(),
            ResidualBlock(depth, depth),
            nn.ReLU(),
            ResidualBlock(depth, depth),
        )

    def forward(self, inputs):
        out = self.initial_linear(inputs)
        out = F.relu(out.view(-1, *self.initial_conv_shape))
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            out = layer(out)
        return out


class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim=50, num_layers=4, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = 35

        self.fc = nn.Linear(feature_dim, num_filters * self.out_dim * self.out_dim)

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, obs_shape[0], 3, stride=2, output_padding=1)
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs["fc"] = h

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        self.outputs["deconv1"] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs["deconv%s" % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs["obs"] = obs

        return obs
