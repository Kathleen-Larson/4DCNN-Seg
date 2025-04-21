import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from collections import OrderedDict

import utils

"""
This file contains the main class (UCNetLong) for segmentation (unet) and 
classification (cnet) of longitudinal data. The unet and cnet portions are 
each built in a separate function within UCNetLong (UCNetLong._build_unet and 
UCNetLong._build_cnet). They are then stored separately as self.unet and 
self.cnet, allowing the user to either train both simultaneously, or freeze 
the weights of one of one while continuing to train the other. The components 
of each model are built using several different classes:
- _ConvBlock: calls a series of _ConvLayers and handes residual/skip conns.
- _ConvLayer: single convolution layer (conv + norm + activ + drop)
- _CNetFlattenBlock: prepares data for last linear layers of class. network
- _CNetLinearBlock: final block of class. network (conv + activation + linear)

There are also several classes based on 2/3d functions in torch.nn that have
been adapted to handle 6d data ([B, C, T, H, W, D]).
- _DownConv: downsampling via 4D convs (or just 2/3d convs for each timepoint)
- _UpConv: upsampling via 4D convs (or just 2/3d convs for each timepoint)
- _Pool: pooling function (max or avg, calls nn.MaxPoolXd or nn.AvgPoolXd)
- _Conv4d: adaptation of nn.ConvXd
- _ConvTranspose4d: transpose adaptation of _Conv4d (aka nn.ConvTransposeXd)

In addition, there are several utility functions the end of the file that 
mostly deal with parsing arguments. These are defined separately rather than 
part of the within class __init__ functions to help keep the code clean. These
include:
- _parse_arg_as_list: 
"""

#------------------------------------------------------------------------------

class UCNetLong(nn.ModuleDict):
    """
    Joint unet+classifier change detection network. Optional arguments for the 
    unet and cnet portions can either be specified as "arg_unet/arg_cnet", or
    simply as "arg" (if same parameter for both).
    
    Required inputs:
    - in_channels:        no. input channels (most likely image modalities)
    - out_channels_unet:  no. output channels for unet (label classes)
    - out_channels_cnet:  no. output channels for cnet (total disease classes)

    Optional args (general):
    - feature_ratio:        factor to increase features at each level
    - n_levels:             number of unet/cnet levels
    - n_starting_features:  number of features at highest spatial res.
    - transfer_index:       index of unet decoder output for cnet input 
    - do_class:             flag to perform classification (do cnet arm)
    - do_combine_nets:      flag to combine decoding arm (unet) and cnet arm
    - do_segment:           flag to perform segmentation (do decoding arm)
    - T:                    index of temporal dimension in input image
    - X:                    number of spatial dimensions in input image
    
    Optional args (unet):
    - activ_func_unet           activation function for conv layers
    - conv_kernel_size_unet:    size of 4D conv kernel
    - conv_shape_unet:          shape of 4D conv kernel (hypercube/hypercross)
    - dropout_rate_unet:        dropout rate in conv layers (0 = no dropout)
    - n_convs_per_block_unet:   number of conv layers per unet level
    - norm_func_unet:           normalization function for conv layers
    - sample_down_func_unet:    downsampling function for enc. arm
    - sample_up_func_unet:      upsampling function for dec. arm
    - sample_kernel_size_unet:  size of sampling pool/conv kernel
    - use_residuals:            flag for residual conns. in enc. arm
    - use skips:                flag for skip conns. between enc./dec. arms

    Optional args (cnet):
    - activ_func_cnet           activation function for conv layers
    - conv_size_cnet:           size of 4D conv kernel
    - conv_shape_cnet:          shape of 4D conv kernel (hypercube/hypercross)
    - dropout_rate_cnet:        dropout rate in conv layers (0 = no dropout)
    - n_convs_per_block_cnet:   number of conv layers per cnet level
    - norm_func_cnet:           normalization function for conv layers
    - sample_down_func_cnet:    downsampling function for enc. arm
    - sample_kernel_size_cnet:  size of sampling pool/conv kernel
    """
    
    def __init__(self,
                 in_channels:int,
                 out_channels_unet,
                 out_channels_cnet:int=None,
                 do_class:bool=True,
                 do_combine_nets:bool=False,
                 do_segment:bool=True,
                 n_starting_features:int=64,
                 transfer_index:int=1,
                 use_temporal:str=True,
                 T:int=2,
                 X:int=3,
                 **kwargs
    ):
        super(UCNetLong, self).__init__()
        
        def _parse_kwargs(self, indict:dict, net_type:str):
            # Parse into dict with required naming convention
            arg_keys = [
                'activ_func', 'conv_size', 'conv_shape', 'down_func',
                'dropout_rate', 'feature_ratio', 'n_convs_per_block',
                'n_levels', 'norm_func', 'sample_size'
            ]
            arg_vals = [
                indict[f'{key}_{net_type}'] if f'{key}_{net_type}' in indict
                else indict[key] if key in indict else None
                for key in arg_keys
            ]
            outdict = dict(zip(arg_keys, arg_vals))

            # Conv block args
            outdict['activ_func'] = _parse_arg_as_function(
                self, outdict['activ_func']
            )
            outdict['conv_size'] = _parse_arg_as_list(
                self, outdict['conv_size'], int
            )
            outdict['norm_func'] = _parse_arg_as_function(
                self, outdict['norm_func']
            )
            
            # Down sampling function args
            outdict['pool_type'] = (
                'Avg' if 'avg' in outdict['down_func'].lower()
                else 'Max' if 'max' in outdict['down_func'].lower()
                else None
            )
            outdict['down_func'] = (
                eval('_DownConv') if 'conv' in outdict['down_func'].lower()
                else eval('_Pool')
            )
            outdict['sample_size'] = _parse_arg_as_list(
                self, outdict['sample_size'], int
            )
            return outdict
        
        # Parse general args
        self.do_class = do_class
        self.do_segment = do_segment
        self.do_combine_nets = do_combine_nets
        self.use_temporal = use_temporal
        
        self.L = transfer_index
        self.T = T
        self.X = X
        
        # Make sure required inputs exist
        if self.do_class and out_channels_cnet is None:
            utils.arg_error(
                'out_channels_cnet required if do_class=True', self
            )
        if self.do_segment and out_channels_unet is None:
            utils.arg_error(
                'out_channels_unet required if do_segment=True', self
            )

        # Build unet
        unet_config = _parse_kwargs(self, kwargs, 'unet')
        self.unet = self._build_unet(
            in_channels=in_channels, out_channels=out_channels_unet,
            n_starting_features=n_starting_features,
            **unet_config
        )
        
        # Build cnet
        if self.do_class:
            cnet_config = _parse_kwargs(self, kwargs, 'cnet')
            self.cnet = self._build_cnet(
                in_channels=n_starting_features,
                out_channels=out_channels_cnet, **cnet_config
	    )

        
    #--------------------------------------------------------------------------
    def _build_unet(self,
                    in_channels,
                    out_channels,
                    activ_func:str='ELU',
                    conv_size:list[int]=3,
                    conv_shape:str='hypercube',
                    down_func:str='_Pool',
                    dropout_rate:float=0.,
                    feature_ratio:float=2.,
                    n_convs_per_block:int=2,
                    n_levels:int=3,
                    n_starting_features:int=24,
                    norm_func:str='Instance',
                    pool_type:str=None,
                    sample_size:list[int]=2,
                    use_residuals:bool=False,
                    use_skips:bool=True,
    ):
        """
        Build the unet portion (encoding/decoding arms) of the network.
        """
        
        # Initialize
        model = nn.Module()
        model.n_levels = n_levels
        model.use_residuals = use_residuals
        model.use_skips = use_skips
    
        feature_config = [
            int(n_starting_features * (feature_ratio ** n))
            for n in range(n_levels)
        ]
        enc = [in_channels] + feature_config[:-1]
        dec = feature_config

        conv_block_kwargs = {
            'activ_func': activ_func, 'norm_func': norm_func,
            'conv_size': conv_size, 'conv_shape': conv_shape,
            'dropout_rate': dropout_rate, 'residual': use_residuals,
            'temporal': self.use_temporal, 'X': self.X, 'T': self.T
        }
        
        # Encoding arm
        model.enc_block = nn.ModuleList(
            [_ConvBlock(
                feature_config=(
                    [enc[n]] + ([enc[n+1]] * n_convs_per_block)),
                **conv_block_kwargs
            ) for n in range(n_levels - 1)]
        )
        model.downsample = nn.ModuleList(
            [down_func(
                n_features=enc[n+1],
                pool_type=pool_type,
                sample_factor=sample_size,
                temporal=False,
                X=self.X,
                T=self.T
            ) for n in range(n_levels - 1)]
        )
            
        # Bottleneck
        model.bottleneck = _ConvBlock(
            feature_config=(
                [enc[-1]] + [dec[-1]] * n_convs_per_block),
            **conv_block_kwargs
        )
                    
        # Decoding arm
        model.dec_block = nn.ModuleList(
            [_ConvBlock(
                feature_config=(
                    [dec[n+1] + enc[n+1] if model.use_skips else dec[n+1]]
                    + ([dec[n]] * n_convs_per_block)),
                **conv_block_kwargs
            ) for n in range(n_levels - 1)]
        )
        model.upsample = nn.ModuleList(
            [_UpConv(
                n_features=dec[n+1],
                sample_factor=sample_size,
                temporal=False,
                X=self.X,
                T=self.T
            ) for n in range(n_levels - 1)]
        )

        # Final conv block
        for key in ['activ_func', 'dropout_rate', 'norm_func']:
            conv_block_kwargs.pop(key)
        conv_block_kwargs['conv_size'] = [1] * (self.X + 1)
            
        model.final = _ConvBlock(
            feature_config=[dec[0], out_channels], **conv_block_kwargs
        )
        return model
                    

    #--------------------------------------------------------------------------
    def _build_cnet(self,
                    in_channels,
                    out_channels,
                    activ_func:str='ELU',
                    conv_size:list[int]=3,
                    conv_shape:str='hypercube',
                    down_func:str='_Pool',
                    dropout_rate:float=0.,
                    feature_ratio:float=2.,
                    n_convs_per_block:int=2,
                    n_levels:int=3,
                    n_linears:int=3,
                    norm_func:str='Instance',
                    pool_type:str='Max',
                    sample_size:list[int]=2,
    ):
        """
        Build the cnet portion (encoding/decoding arms) of the network.
        """
        # Initialize
        model = nn.Module()
        model.n_levels = n_levels
        
        cls = [
            int(in_channels * (feature_ratio ** n)) for n in range(n_levels)
        ]
        
        # Down sampling
        model.conv_block = nn.ModuleList(
            [_ConvBlock(
                activ_func=activ_func,
                conv_size=conv_size,
                conv_shape=conv_shape,
                dropout_rate=dropout_rate,
                feature_config=(
                    [cls[n]] + ([cls[n+1]] * n_convs_per_block)),
                norm_func=norm_func,
                temporal=self.use_temporal,
                X=self.X,
                T=self.T
            ) for n in range(n_levels - 1)]
        )
        model.downsample = nn.ModuleList(
            [down_func(
                n_features=cls[n+1],
                sample_factor=sample_size,
                temporal=False,
                X=self.X,
                T=self.T,
            ) for n in range(n_levels - 1)]
        )
        
        # Final linear block
        model.flatten = _CNetFlattenBlock(
            n_features=cls[-1],
            kernel_size=sample_size[0],
            T=self.T,
        )
        model.final = _CNetLinearBlock(
            feature_config=(cls[::-1] + [out_channels]),
            activ_func=activ_func,
            conv_size=conv_size[0]
        )

        return model
        
    
    #--------------------------------------------------------------------------
    def forward(self, x):
        """
        Foward call
        """

        # Storage for network outputs
        skips = [None] * (self.unet.n_levels - 1)
        siz = [None] * (self.unet.n_levels - 1)

        # UNet
        for n in range(self.unet.n_levels - 1):
            x = self.unet.enc_block[n](x)
            if self.unet.use_skips:
                skips[n], siz[n] = (x, x.shape)
            x = self.unet.downsample[n](x)

        x = self.unet.bottleneck(x)

        for n in reversed(range(self.unet.n_levels - 1)):
            x = self.unet.upsample[n](x, siz[n])
            x = self.unet.dec_block[n](
                torch.cat([skips[n], x], dim=1) if self.unet.use_skips else x
            )
            
        unet_logits = self.unet.final(x)
        
        if not self.do_class:
            return unet_logits, None

        # CNet
        if not self.do_combine_nets:
            for n in range(self.cnet.n_levels - 1):
                x = self.cnet.downsample[n](self.cnet.conv_block[n](x))
            x = self.cnet.flatten(x)
            
        cnet_logits = self.cnet.final(x)
        return unet_logits, cnet_logits



    
#-----------------------------------------------------------------------------#
#                                Unet functions                               #
#-----------------------------------------------------------------------------#

class _ConvBlock(nn.ModuleDict):
    def __init__(self,
                 feature_config:list[int],
                 activ_func=None,
                 conv_size:list[int]=3,
                 conv_shape:str='hypercube',
                 dropout_rate:float=0.,
                 norm_func=None,
                 residual:bool=False,
                 temporal:bool=False,
                 X:int=3,
                 T:int=2
    ):
        super(_ConvBlock, self).__init__()
        self.use_residuals = residual

        # Parse args
        n_layers = len(feature_config) - 1
        self.use_residuals = residual
        
        for n in range(n_layers):
            layer = _ConvLayer(
                n_input_features=feature_config[n],
                n_output_features=feature_config[n+1],
                activ_func=activ_func,
                conv_size=conv_size,
                conv_shape=conv_shape,
                dropout_rate=dropout_rate,
                norm_func=norm_func,
                temporal=temporal,
                X=X,
                T=T
            )
            self.add_module(f'ConvLayer{n+1}', layer)
            

    def forward(self, x):
        for n, [name, layer] in enumerate(self.items()):
            res = x
            x = layer(x) + res if (
                self.use_residuals and name[-1]!='1'
            ) else layer(x)
            
        return x


#-----------------------------------------------------------------------------
    
class _ConvLayer(nn.Module):
    def __init__(self,
                 n_input_features:int,
                 n_output_features:int,
                 activ_func=None,
                 conv_size:list[int]=3,
                 conv_shape:str='hypercube',
                 dropout_rate:float=0.,
                 norm_func=None,
                 temporal:bool=True,
                 X:int=3,
                 T:int=2
    ):
        super(_ConvLayer, self).__init__()

        # Parse args
        conv_kernel_size = (
            [conv_size] * X if isinstance(conv_size, int) else conv_size
        )
        conv_padding_size = [
            ((cs - 1) // 2) if cs % 2 == 1 else (cs // 2)
            for cs in conv_kernel_size
        ]
        self.temporal = temporal
        self.T = T
        self.X = X
        
        # Conv function
        self.conv = _Conv4d(
            in_channels=n_input_features,
            out_channels=n_output_features,
            kernel_size=conv_kernel_size,
            kernel_shape=conv_shape,
            padding=conv_padding_size,
            bias=False if norm_func is not None else True
        ) if self.temporal else (
            nn.ModuleList(
                [eval(f'nn.Conv{X}d')(
                    in_channels=n_input_features,
                    out_channels=n_output_features,
                    kernel_size=conv_kernel_size[-X:],
                    padding=conv_padding_size[-X:],
                    bias=False if norm_func is not None else True
                ) for t in range(self.T)]
            )
        )

        # Norm + activation + dropout
        self.layer_funcs = nn.ModuleList([nn.Sequential(
            OrderedDict([
                ('norm', norm_func(n_output_features)
                 if norm_func is not None else nn.Identity()),
                ('activ', activ_func()
                 if activ_func is not None else nn.Identity()),
                ('drop', eval(f'nn.Dropout{X}d')(p=dropout_rate)
                 if dropout_rate > 0. else nn.Identity())
            ])
        ) for t in range(self.T)])
        

    def forward(self, x):
        x = torch.stack(
            [self.layer_funcs[t](xt)
             for t, xt in enumerate(self.conv(x).movedim(-(self.X+1), 0))
            ] if self.temporal else
            [self.layer_funcs[t](self.conv[t](xt))
             for t, xt in enumerate(x.movedim(-(self.X+1), 0))
            ], dim=-(self.X+1)
        )
        return x



#-----------------------------------------------------------------------------#
#                            Classifier functions                             #
#-----------------------------------------------------------------------------#

class _CNetFlattenBlock(nn.Module):
    """
    Takes an input from w/ size [B, C, T, H, W, D] and prepares for input to 
    linear layer of classifier
    1. Flatten spatial dimensions --> [B, C, T, (H x W x D)]
    2. Downsampling temporal dim w/ 2D conv --> [B, C, 1, (H x W x D)]
    """
    def __init__(self,
                 n_features:int,
                 kernel_size:int,
                 pool_type:str='Avg',
                 X:int=3,
                 T:int=2,
    ):
        super(_CNetFlattenBlock, self).__init__()
        self.X = X
        self.T = T

        self.spatial_pool = eval(f'nn.Adaptive{pool_type}Pool{X}d')(
            (1,) * self.X
        )
        self.temporal_downsample = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True
        )
        
        
    def forward(self, x):
        x = self.temporal_downsample(
            torch.stack(
                [self.spatial_pool(xt)
                 for xt in x.movedim(-(self.X+1), 0)
                ], dim=-(self.X+1)
            ).view(*x.shape[:-self.X])
        ).squeeze(-1)
        return x

    
#-----------------------------------------------------------------------------
    
class _CNetLinearBlock(nn.ModuleDict):
    """
    Takes a fully downsampled input and produces the final classifier output
    """
    def __init__(self,
                 feature_config:list[int],
                 activ_func=None,
                 conv_size:int=3,
    ):
        super(_CNetLinearBlock, self).__init__()
        self.n_layers = len(feature_config) - 1
        padding_size = (
            (conv_size - 1) // 2 if conv_size % 2 == 1 else conv_size // 2
        )

        # Add intermediate layers
        for n in range(self.n_layers - 1):
            self.add_module(
                f'LinearLayer{n+1}', nn.Sequential(
                    nn.Linear(
                        in_features=feature_config[n],
                        out_features=feature_config[n+1],
                    ),
                    activ_func() if activ_func is not None else nn.Identity()
                )
            )

        # Add last linear function
        self.LastLinear = nn.Linear(
            in_features=feature_config[-2],
            out_features=feature_config[-1],
        )
        
    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, list) else x

        for n in range(self.n_layers - 1):
            x = self.__getattr__(f'LinearLayer{n+1}')(x)
        x = self.LastLinear(x)

        return x

    

#-----------------------------------------------------------------------------#
#                          Functions adapted for 4D                           #
#-----------------------------------------------------------------------------#

class _DownConv(nn.Module):
    def __init__(self,
                 n_features:int,
                 sample_factor:list[int],
                 sample_shape:str='hypercube',
                 temporal:bool=True,
                 X:int=3,
                 T:int=2,
                 **kwargs
    ):
        super(_DownConv, self).__init__()

        # Parse class args
        self.T = T
        self.temporal = temporal

        # Define sample function
        self.downsample = _Conv4d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_shape=sample_shape,
            kernel_size=sample_factor,
            stride=sample_factor,
            bias=False,
            X=X,
            T=T,
        ) if self.temporal else nn.ModuleList(
            [eval(f'nn.Conv{X}d')(
                in_channels=n_input_features,
                out_channels=n_output_features,
                kernel_size=sample_factor[-X:],
                stride=sample_factor[-X:],
                bias=True,
            ) for t in range(self.T)]
        )

        
    def forward(self, x, db=False):
        x = self.downsample(x) if self.temporal else torch.stack(
            [self.downsample[t](xt)
             for t, xt in enumerate(x.movedim(-(self.X+1), 0))
            ], dim=-(self.X+1)
        )
        return x


#-----------------------------------------------------------------------------
    
class _UpConv(nn.Module):
    def __init__(self,
                 n_features:int,
                 sample_factor:list[int],
                 sample_shape:str='hypercube',
                 temporal:bool=True,
                 X:int=3,
                 T:int=2,
    ):
        super(_UpConv, self).__init__()

        # Parse class args
        self.sample_factor = sample_factor
        self.temporal = temporal
        self.T = T
        self.X = X

        # Define sample function
        self.upsample = _Conv4dTranspose(
            in_channels=n_features,
            out_channels=n_features,
            kernel_shape=sample_shape,
            kernel_size=sample_factor,
            bias=False,
            X=X,
            T=T,
        ) if self.temporal else (
            nn.ModuleList(
                [eval(f'nn.ConvTranspose{X}d')(
                    in_channels=n_features,
                    out_channels=n_features,
                    kernel_size=sample_factor[-X:],
                    stride=sample_factor[-X:],
                    bias=True,
                ) for t in range(self.T)]
            )
        )
    

    def forward(self, x, siz):
        x = self.upsample(x) if self.temporal else torch.stack(
            [self.upsample[t](xt)
             for t, xt in enumerate(x.movedim(-(self.X+1), 0))
            ], dim=-(self.X+1)
        )
        return x


#------------------------------------------------------------------------------

class _Pool(nn.Module):
    def __init__(self,
                 pool_type:str='Max',
                 sample_factor:list[int]=2,
                 temporal:bool=True,
                 X:int=3,
                 T:int=2,
                 **kwargs
    ):
        super(_Pool, self).__init__()

        # Parse class args
        self.temporal = temporal
        self.T = T
        self.X = X

        # Define pooling functions
        self.spatial_pool = eval(f'nn.{pool_type}Pool{X}d')(
            kernel_size=sample_factor[1:],
            stride=sample_factor[1:],
        )
        self.temporal_pool = eval(f'nn.{pool_type}Pool2d')(
            kernel_size=(sample_factor[0], 1),
            stride=(sample_factor[0], 1),
        ) if self.temporal else None
        

    def forward(self, x):
        nB, nC, nT = x.shape[:self.T+1]
        nvox = np.prod(x.shape[self.T+1:])

        x_pool = torch.stack(
            [self.spatial_pool(xt)
             for xt in x.movedim(-(self.X+1), 0)
            ], dim=-(self.X+1)
        )
        x_pool = self.temporal_pool(
            x_pool.view(nB, nC, nT, nvox)
        ).view(nB, nC, nT, *x.shape[-self.X:]) if self.temporal else x_pool
        
        return x_pool
    
        
#------------------------------------------------------------------------------

class _Conv4d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:[int, tuple]=3,
                 stride:[int, tuple]=1,
                 padding:[int, tuple]=0,
                 dilation:[int, tuple]=1,
                 groups=1,
                 bias=False,
                 padding_mode:str='zeros',
                 kernel_shape='hypercube',
                 layer_func:str='Conv',
                 X:int=3,
                 T:int=2
    ):
        super(_Conv4d, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Parse class attributes
        self.X = X
        self.T = T
        
        self.kernel_size = _parse_arg_as_list(self, kernel_size, int)
        self.stride = _parse_arg_as_list(self, stride, int)
        self.padding = _parse_arg_as_list(self, padding, int)
        self.dilation = _parse_arg_as_list(self, dilation, int)
        
        valid_kernel_shapes = ['hypercube', 'hypercross']
        kernel_shape = _valid_configured_input(
            self, kernel_shape, valid_kernel_shapes
        )

        valid_padding_modes = ['zeros']
        padding_mode = _valid_configured_input(
            self, padding_mode, valid_padding_modes
        )

        layer_func = _parse_arg_as_function(self, layer_func)
        
        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(
            torch.Tensor(
                out_channels, in_channels // groups, *self.kernel_size
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()
        
        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv_layers = torch.nn.ModuleList()
        
        for i in range(self.kernel_size[0]):
            if kernel_shape == 'hypercube':
                kernel_size_i = self.kernel_size[1::]
            elif kernel_shape == 'hypercross':
                kernel_size_i = (
                    self.kernel_size[1::]
                    if i in [(self.kernel_size[0] - 1)//2]
                    else 1
                )
            layer = layer_func(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size_i,
                padding=self.padding[1::],
                dilation=self.dilation[1::],
                stride=self.stride[1::],
                bias=bias
            )
            layer.weight = nn.Parameter(self.weight[:, :, i, :, :])
            self.conv_layers.append(layer)
            
        del self.weight


    def _get_output_dim(self, I, k, p, d, s):
        return ((I + (2 * p) - k - ((d - 1) * (k - 1))) // s) + 1


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, x):
        # Get size of output temporal dim
        kT = self.kernel_size[0]
        pT = self.padding[0]
        dT = self.dilation[0]
        sT = self.stride[0]

        Ti = x.shape[self.X-1]
        To = self._get_output_dim(Ti, kT, pT, dT, sT)        

        # Build output by iterating over temporal kernel size
        out = [None] * To
        
        for n in range(kT):
            # Get range of frame_j w.r.t frame_n
            zero_offset = -pT + (n * dT)
            j_start = max(zero_offset % sT, zero_offset)
            j_end = min(Ti, Ti + pT - (dT * (kT-n-1)))

            # Convolve frame_n with frame_j
            for j in range(j_start, j_end, sT):
                out_frame = (j - zero_offset) // sT
                if out[out_frame] is None:
                    out[out_frame] = self.conv_layers[n](x[:, :, j, ...])
                else:
                    out[out_frame] += self.conv_layers[n](x[:, :, j, ...])
                    
        # Reshape and add bias
        out = torch.stack(out, dim=(self.X-1))
        if self.bias is not None:
            out += self.bias.view((1, -1) + (1,) * (self.X+1))

        return out


    
class _ConvTranspose4d(_Conv4d):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            layer_func='ConvTranspose',
            **kwargs
        )

    def _get_output_dim(self, I, k, p, d, s):
        return (s * (I - 1)) - (d * (k - 1)) - (2 * d) + 1
        
    
#-----------------------------------------------------------------------------#
#                              Utility functions                              #
#-----------------------------------------------------------------------------#

def _parse_arg_as_list(cls, arg, dtype):
    varname = f'{arg=}'.split('=')[0]
    if not isinstance(arg, list):
        arg = (list(arg) if isinstance(arg, tuple)
               else [arg] * (cls.X+1)
        )
    if len(arg) != cls.X + 1:
        if len(arg) == 1:
            arg = arg * (cls.X+1)
        else:
            varname = f'{arg=}'.split('=')[0]
            utils.arg_error(
                f'{varname} must be {dtype} or list/tuple of {dtype} of '
                f'len=={cls.X+1} (input was {arg})', cls
            )
    return arg

        
def _parse_arg_as_function(cls, arg):
    possible_funcs = [
        f'{arg}', f'nn.{arg}', f'{arg}{cls.X}d', f'nn.{arg}{cls.X}d'
    ]
    found = False
    for func in possible_funcs:
        try:
            arg = eval(func)
            found = True
        except:
            pass        
    if not found:
        varname = f'{arg=}'.split('=')[0]
        utils.arg_error(
            f'{arg} (or {arg}{cls.X}d) is not a valid input for {varname} '
            f'(not a valid callable function or an attribute of torch.nn)'
            , cls
        )
    return arg

    
def _valid_configured_input(cls, arg, valid_list):
    if arg not in valid_list:
        varname = f'{arg=}'.split('=')[0]
        utils.arg_error(
            f'{arg} not a valid input for {varname} (must be in '
            f'{valid_list})', cls
        )
    return arg

        

def _print_model_size(model):
    param_size = 0.
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0.
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'model size: {size_all_mb:.3f}MB')
    
    
