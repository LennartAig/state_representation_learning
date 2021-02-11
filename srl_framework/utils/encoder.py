import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from srl_framework.utils.networks import make_cnn, inits

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

def compare_weights(src, trg):
    assert type(src) == type(trg)
    return np.abs(np.sum(trg.weight.data.cpu().numpy() - src.weight.data.cpu().numpy()))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            Conv2dSame(in_channels, out_channels, 3)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class CnnEncoder(nn.Module):
    """
    - TODO
    """
    def __init__(self, img_channels = 3, feature_dim=128, img_size = 84, architecture='std2', normalized_obs=True,
                squash_latent = True, normalize = True, activation="ReLU", batchnorm=False, pool=False,
                dropout = False, conv_init="delta_orthogonal", linear_init="orthogonal"):
        super(CnnEncoder, self).__init__()

        self.squash_latent = squash_latent
        self.normalize = normalize
        self.batchnorm = batchnorm
        self.normalized_obs = normalized_obs

        if architecture == 'std':
            OUT_DIM = {128: 46, 84: 24, 64: 14}
            channels = [img_channels] + [32,32,32,32,32,32]
            kernel_sizes = [6,5,5,5,3,3]
            stride_size = [2,1,1,1,1,1]
            padding_sizes= [0,0,0,0,0,0]
            self.depth = 6
        elif architecture == 'std2':
            OUT_DIM = {128: 41, 84: 31, 64: 21}
            channels = [img_channels] + [32,32,32,32,32,32]
            kernel_sizes = [3,3,3,3,3,3]
            stride_size = [2,1,1,1,1,1]
            padding_sizes= [0,0,0,0,0,0]
            self.depth = 6
        elif architecture == 'nature':
            assert (img_size >= 84),"Image Size must be at least 128, should be higher with nature cnn!"
            OUT_DIM = {128:14, 84: 7}
            self.normalize = False 
            channels = [img_channels] + [32,64,128,64]
            kernel_sizes = [8,4,4,3]
            stride_size = [3,2,1,1]
            padding_sizes= [0,0,0,0]
            self.depth = 4
        else:
            raise NotImplementedError
        self.depths = channels
        self.out_dim = OUT_DIM[img_size]         
        h_dim = channels[-1]*self.out_dim*self.out_dim
        
        self.encoder = make_cnn(channels = channels, kernels=kernel_sizes, strides=stride_size, paddings=padding_sizes,
            activation=activation, batchnorm=batchnorm, pool=pool, dropout = dropout, conv_init=conv_init)
        self.output_layer = nn.Linear(h_dim, feature_dim).apply(inits[linear_init])
        self.flatten = Flatten()
        if self.normalize: self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x, fmaps=False, detach = False):
        if not self.normalized_obs:
            x = x/255.0
        if self.batchnorm:
            f5 = self.encoder[:((self.depth-1)*3)](x)
            f7 = self.encoder[((self.depth-1)*3):](f5)
        else:
            f5 = self.encoder[:((self.depth-1)*2)](x)
            f7 = self.encoder[((self.depth-1)*2):](f5)
        out = self.flatten(f7)
        if detach:
            out = out.detach()
        out = self.output_layer(out)
        out = self.layer_norm(out) if self.normalize else out
        out = F.tanh(out) if self.squash_latent else out
        if fmaps:
            return {
                'f5': f5,
                'out': out
            }
        return out

    @property
    def local_layer_depth(self):
        return self.depths[-2]

    @property
    def output_dim(self):
        return self.out_dim
    
    
    def copy_conv_weights_from(self, source):
        i = 0
        for module_src, module_targ in zip(source.encoder.modules(),self.encoder.modules()):
            if type(module_targ) == nn.Conv2d:
                tie_weights(src=module_src, trg=module_targ)
            if type(module_targ) != nn.Sequential:
                i += 1


class ResNetEncoder(nn.Module):
    def __init__(self, img_channels = 3, feature_dim=128, img_size = 84, normalized_obs=True,
                squash_latent = True, normalize = True, conv_init="delta_orthogonal", linear_init="orthogonal"):
        super(ResNetEncoder, self).__init__()

        self.squash_latent = squash_latent
        self.normalize = normalize
        self.normalized_obs = normalized_obs
        
        self.depths = [16, 32, 32, 32]
        OUT_DIM = {128: 7, 84: 4, 64: 3}
        self.layer1 = self._make_layer(img_channels, self.depths[0]).apply(inits[conv_init])
        self.layer2 = self._make_layer(self.depths[0], self.depths[1]).apply(inits[conv_init])
        self.layer3 = self._make_layer(self.depths[1], self.depths[2]).apply(inits[conv_init])
        self.layer4 = self._make_layer(self.depths[2], self.depths[3]).apply(inits[conv_init])

        self.out_dim = OUT_DIM[img_size]         
        self.final_conv_size = self.depths[-1]*self.out_dim*self.out_dim
        self.output_layer = nn.Linear(self.final_conv_size, feature_dim).apply(inits[linear_init])
        self.flatten = Flatten()
        if self.normalize: self.layer_norm = nn.LayerNorm(feature_dim)

    def _make_layer(self, in_channels, depth):
        return nn.Sequential(
            Conv2dSame(in_channels, depth, 3),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(),
            ResidualBlock(depth, depth),
            nn.ReLU(),
            ResidualBlock(depth, depth)
        )

    @property
    def local_layer_depth(self):
        return self.depths[-2]
    
    @property
    def output_dim(self):
        return self.out_dim

    def forward(self, x, fmaps=False, detach=False):
        if not self.normalized_obs:
            x = x/255.0
        f5 = self.layer3(self.layer2(self.layer1(x)))
        f7 = self.layer4(f5)
        out = self.output_layer(self.flatten(f7))
        out = out if not self.normalize else self.layer_norm(out)
        out = out if self.output_logits else F.tanh(out)
        out = out if self.output_relu else F.relu(out)
        if fmaps:
            return {
                'f5': f7,
                'out': out
            }
    
    def copy_conv_weights_from(self, source):
        i = 0
        for module in self.encoder.modules():
            if type(module) == nn.Conv2d:
                tie_weights(src=source.encoder[i], trg=self.encoder[i])
            if type(module) != nn.Sequential:
                i += 1


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, vae=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.vae = vae
        self.feature_dim = 2*feature_dim if self.vae else feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = 35
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h
    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = torch.tanh(h_norm)
        self.outputs['tanh'] = out
        if self.vae:
            mu, logvar = out.chunk(2, dim=-1)
            self.outputs['mu'] = mu
            self.outputs['logvar'] = logvar

            out = self.reparameterize(mu, logvar)

        return out
    def forward_latent(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = torch.tanh(h_norm)
        self.outputs['tanh'] = out
        if self.vae:
            mu, logvar = out.chunk(2, dim=-1)
            self.outputs['mu'] = mu
            self.outputs['logvar'] = logvar

            out = self.reparameterize(mu, logvar)

        return out, mu, logvar
    
    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])
    
    def compare_conv_weights_from(self, source):
        diff_conv = 0
        diff_lin = 0
        for i in range(self.num_layers):
            diff_conv += compare_weights(src=source.convs[i], trg=self.convs[i])
        diff_lin += compare_weights(src=source.fc, trg=self.fc)
        return diff_lin, diff_conv
        #assert diff_conv < 0.1