import torch
import torch.nn as nn
import math

def make_mlp(input_dim, output_dim, architecture= [64,64], activation = "ReLU", output_layer=True,
            batchnorm=False, dropout=False, init = "orthogonal"):
    layer_sizes = [input_dim]+ architecture +[output_dim]
    output_size = layer_sizes[-2] 
    layers = [mlp_layer(in_, 
                    out_, 
                    activation_=activation,
                    batchnorm=batchnorm,
                    dropout=dropout) for in_, out_ in zip(layer_sizes[:-2], layer_sizes[1:-1])] 
    if output_layer:
        output_size = layer_sizes[-1]
        layers.append(mlp_layer(layer_sizes[-2], layer_sizes[-1]))
    return unwrap_layers(nn.Sequential(*layers)), output_size

def make_cnn(channels, kernels, strides, paddings, activation="ReLU", batchnorm=False, pool=False,
            dropout = False, conv_init="delta_orthogonal"):
    layers = []
    for in_, out_, ker_, stride_, pad_ in zip(channels[:-1], channels[1:], kernels, strides, paddings): 
        layers.append(conv_layer(in_,
                                out_,
                                ker_,
                                stride_,
                                pad_,
                                activation=activation,
                                batchnorm=batchnorm,
                                dropout=dropout, pool=pool))#.apply(inits[conv_init]))
    cnn = unwrap_layers(nn.Sequential(*layers))
    return cnn

def make_decoder(channels, kernels, strides, paddings, activation="ReLU", batchnorm=False, pool=False,
            dropout = False, conv_init="delta_orthogonal"):
    layers = []
    for in_, out_, ker_, stride_, pad_ in zip(channels[:-2], channels[1:-1], kernels[:-1], strides[:-1], paddings[:-1]): 
        layers.append(deconv_layer(in_,
                                out_,
                                ker_,
                                stride_,
                                pad_,
                                activation=activation,
                                batchnorm=batchnorm,
                                dropout=dropout, pool=pool))#.apply(inits[conv_init]))
    # final layer 
    layers.append(deconv_layer(channels[-2],channels[-1],kernels[-1],strides[-1],paddings[-1],
                                activation=None,
                                batchnorm=None,
                                dropout=None, pool=False))#.apply(inits[conv_init]))
    cnn = unwrap_layers(nn.Sequential(*layers))
    return cnn

def mlp_layer(in_, out_, activation_=None, dropout=None, batchnorm=False):
    l = nn.ModuleList([nn.Linear(in_, out_)])
    if batchnorm:
        l.append(nn.BatchNorm1d(out_))
    if activation_ is not None:
        activation = getattr(nn.modules.activation, activation_)()
        l.append(activation)
    if dropout:
        l.append(nn.Dropout()) 
    return l

def conv_layer(in_, out_, ker_, stride_, pad_, bias=True, activation=nn.ReLU(), batchnorm=False, dropout=None, pool = False):
    l = nn.ModuleList([nn.Conv2d(in_,out_,kernel_size=ker_,stride=stride_,padding=pad_, bias=bias)])
    if batchnorm:
        l.append(nn.BatchNorm2d(out_))
    if activation is not None:
        if activation == 'LeakyReLU':
            activation =nn.LeakyReLU(0.02)
        else:
            activation = getattr(nn.modules.activation, activation)()
        l.append(activation)
    if dropout:
        l.append(nn.Dropout())
    if pool:
        l.append(nn.MaxPool2d(kernel_size=3, stride=2))
    return l

def deconv_layer(in_, out_, ker_, stride_, pad_, bias=True, activation=nn.ReLU(), batchnorm=False, dropout=None, pool = False):
    l = nn.ModuleList([nn.ConvTranspose2d(in_,out_,kernel_size=ker_,stride=stride_,output_padding=pad_, bias=bias)])
    if batchnorm:
        l.append(nn.BatchNorm2d(out_))
    if activation is not None:
        if activation == 'LeakyReLU':
            activation =nn.LeakyReLU(0.02)
        else:
            activation = getattr(nn.modules.activation, activation)()
        l.append(activation)
    if dropout:
        l.append(nn.Dropout())
    if pool:
        l.append(nn.MaxPool2d(kernel_size=3, stride=2))
    return l

def unwrap_layers(model):
    l = []
    def recursive_wrap(model):
        for m in model.children():
            if isinstance(m, nn.Sequential): recursive_wrap(m)
            elif isinstance(m, nn.ModuleList): recursive_wrap(m)
            else: l.append(m)
    recursive_wrap(model)
    return nn.Sequential(*l)

def naive(m):
    if isinstance(m, nn.Linear):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        nn.init.uniform_(m.weight, a=-math.sqrt(1.0 / float(fan_in)), b=math.sqrt(1.0 / float(fan_in)))
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        nn.init.uniform_(m.weight, a=-math.sqrt(1.0 / float(fan_in)), b=math.sqrt(1.0 / float(fan_in)))

def xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

def kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

def orthogonal(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        try:
            nn.init.zeros_(m.bias)
        except:
            pass

def delta_orthogonal(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


inits = {
    'naive' : naive,
    'xavier': xavier,
    'kaiming': kaiming,
    'orthogonal': orthogonal,
    'delta_othogonal': delta_orthogonal
}