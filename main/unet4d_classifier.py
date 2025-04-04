import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from conv4d import Conv4d, ConvTranspose4d
import utils


#------------------------------------------------------------------------------

class UClassNetXD_Longitudinal(nn.Module):
    """
    Main class for the unet+classifier change detection network.
    
    Required inputs:
    - in_channels:        no. input channels (most likely image modalities)
    - out_channels_unet:  no. output channels for unet (label classes)
    - out_channels_cnet:  no. output channels for cnet (total disease classes)

    Optional args (general):
    - feature_ratio:        factor to increase features at each level
    - n_levels:             number of unet/cnet levels
    - n_starting_features:  number of features at highest spatial res.
    - transfer_index:       index of unet decoder output for cnet input 
    - T:                    index of temporal dimension in input image
    - X:                    number of spatial dimensions in input image
    
    Optional args (unet):
    - activ_func_unet           activation function for conv layers
    - conv_kernel_size_unet:    size of 4D conv kernel
    - conv_shape_unet:          shape of 4D conv kernel (hypercube/hypercross)
    - drop_rate_unet:           dropout rate in conv layers (0 = no dropout)
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
    - drop_rate_cnet:           dropout rate in conv layers (0 = no dropout)
    - n_convs_per_block_cnet:   number of conv layers per cnet level
    - norm_func_cnet:           normalization function for conv layers
    - sample_down_func_cnet:    downsampling function for enc. arm
    - sample_kernel_size_cnet:  size of sampling pool/conv kernel
    """
    def __init__(self,
                 in_channels:int,
                 out_channels_cnet:int,
                 out_channels_unet:int,
                 activ_func_cnet:str='ReLU',
                 activ_func_unet:str='ReLU',
                 conv_kernel_size_cnet:int=3,
                 conv_kernel_size_unet:int=3,
                 conv_shape_cnet:str='hypercube',
                 conv_shape_unet:str='hypercube',
                 dropout_rate_cnet:float=0.,
                 dropout_rate_unet:float=0.,
                 feature_ratio:float=2.,
                 n_convs_per_block_cnet:int=2,
                 n_convs_per_block_unet:int=2,
                 n_levels:int=3,
                 n_starting_features:int=24,
                 norm_func_cnet:str='Instance',
                 norm_func_unet:str='Instance',
                 sample_func_down_cnet:str='_DownSample',
                 sample_func_down_unet:str='_DownSample',
                 sample_func_up_unet:str='_UpSample',
                 sample_kernel_size_cnet:int=2,
                 sample_kernel_size_unet:int=2,
                 transfer_index:int=2,
                 use_residuals:bool=False,
                 use_skips:bool=True,
                 use_temporal:str=True,
                 T:int=2,
                 X:int=3,
    ):
        super(UClassNetXD_Longitudinal, self).__init__()
        
        # Within-class utilities to parse args 
        def _print_arg_error(instr:str):
            utils.fatal(
                f'Error in UClassNetXD_Longitudinal.__init__: {instr}'
            )

        def _arg_to_list(arg, dtype):
            arg = (
                [arg] * (self.X + 1) if isinstance(arg, dtype)
                else [x for x in arg]
            )
            varname = f'{arg=}'.split('=')[0]
            return arg if len(arg) == (self.X + 1) else _print_arg_error(
                f'{varname} must be {dtype} or list/tuple of {dtype} of '
                f'len=={self.X+1} (input was {arg})'
            )

        def _get_function(arg):
            try:
                return eval(f'{arg}')
            except:
                try:
                    return eval(f'{arg}{self.X}d')
                except:
                    varname = f'{arg=}'.split('=')[0]
                    _print_arg_error(
                        f'{arg} (or {arg}{self.X}d) is not a valid input for '
                        f'{varname} (not a valid callable function)'
                    )

        def _valid_configured_input(arg, valid_list):
            if arg not in valid_list:
                varname = f'{arg=}'.split('=')[0]
                _print_arg_error(
                    f'{arg} not a valid input for {varname} (must be in '
                    f'{valid_list})')
            
        # Parse class attributes
        self.use_residuals = use_residuals
        self.use_skips = use_skips
        self.use_temporal = use_temporal
        self.L = transfer_index
        self.T = T
        self.X = X

        feature_config = [
            n_starting_features * (feature_ratio ** n) for n in range(n_levels)
        ]
        self.n_blocks = n_levels
        
        # Make sure other args are correct type/shape
        activ_func_cnet = _get_function(activ_func_cnet)
        activ_func_unet = _get_function(activ_func_unet)

        conv_kernel_size_cnet = _arg_to_list(conv_kernel_size_cnet, int)
        conv_kernel_size_unet = _arg_to_list(conv_kernel_size_unet, int)

        norm_func_cnet = _get_function(norm_func_cnet)
        norm_func_unet = _get_function(norm_func_unet)

        sample_func_down_cnet = _get_function(sample_func_down_cnet)
        sample_func_down_unet = _get_function(sample_func_down_unet)
        sample_func_up_unet = _get_function(sample_func_up_unet)

        sample_kernel_size_cnet = _arg_to_list(sample_kernel_size_cnet, int)
        sample_kernel_size_unet = _arg_to_list(sample_kernel_size_unet, int)
        
        valid_conv_shapes = ['hypercube', 'hypercross']
        conv_shape_cnet = (
            conv_shape_cnet if conv_shape_cnet in valid_conv_shapes
            else _print_arg_error(
                    f'{conv_shape_cnet} not a valid input for conv_shape_cnet '
                    f'(must be in {valid_conv_shapes})')
        )
        conv_shape_unet = (
            conv_shape_unet if conv_shape_cnet in valid_conv_shapes
            else _print_arg_error(
                    f'{conv_shape_unet} not a valid input for conv_shape_unet '
                    f'(must be in {valid_conv_shapes})')
        )
                
        # UNet encoding arm:
        self.unet_encoder = nn.Sequential()
        unet_enc_config = [in_channels] + feature_config
        
        for b in range(self.n_blocks):
            block_config = (
                [unet_enc_config[b]]
                + [unet_enc_config[b+1]] * n_convs_per_block_unet
            )
            convblock = _UNetBlock(
                activ_func=activ_func_unet,
                conv_size=conv_kernel_size_unet,
                conv_shape=conv_shape_unet,
                dropout_rate=dropout_rate_unet,
                feature_config=block_config,
                norm_func=norm_func_unet,
                temporal=self.use_temporal,
                X=self.X,
                T=self.T
            )
            downsample = sample_func_down_unet(
                n_features=block_config[-1],
                sampling_factor=sample_kernel_size_unet,
                temporal=False,
                X=self.X,
                T=self.T,
            )
            self.unet_encoder.add_module(f'ConvBlock{b+1}', convblock)
            self.unet_encoder.add_module(f'Downsample{b+1}', downsample)
                            
        # UNet decoding arm:
        self.unet_decoder = nn.Sequential()
        unet_dec_config = [out_channels_unet] + feature_config
        
        for b in reversed(range(self.n_blocks)):
            block_config = (
                [unet_dec_config[b+1]] * n_convs_per_block_unet
                + [unet_dec_config[b]]
            )
            upsample = sample_func_up_unet(
                n_features=block_config[0],
                sampling_factor=sample_kernel_size_unet,
                temporal=False,
                X=self.X,
                T=self.T,
            )
            convblock = _UNetBlockTranspose(
                activ_func=activ_func_unet,
                conv_size=conv_kernel_size_unet,
                conv_shape=conv_shape_unet,
                dropout_rate=dropout_rate_unet,
                feature_config=block_config,
                norm_func=norm_func_unet,
                temporal=self.use_temporal,
                skip=self.use_skips,
                X=self.X,
                T=self.T
            )
            self.unet_decoder.add_module(f'Upsample{b+1}', upsample)
            self.unet_decoder.add_module(f'ConvBlock{b+1}', convblock)

        # Classifier arm
        self.classifier = nn.Sequential()
        n_cnet_features = torch.tensor(cnet_feature_config).sum().item()
        cnet_input_feature_config = unet_dec_config[1:]
        cnet_output_feature_config = (
            [n_cnet_features] * n_convs_per_block_cnet + out_channels_unet
        )        

        ## TO-DO: write these functions
        
        for b in range(self.n_blocks):
            spatial_downsample = _CNetSpatialDownBlock(
                n_features=classifier_config[b],
                sampling_func=sample_func_down_cnet,
                sampling_factor=sample_kernel_size_unet,
                X=self.X,
                T=self.T,
            )
            temporal_downsample = _CNetTemporalDownBlock(
                n_features=classifier_config[b],
                sampling_func=sample_func_down_cnet,
                sampling_factor=sample_kernel_size_unet,
                X=self.X,
                T=self.T,
            )
            self.classifier.add_module(
                f'SpatialBlock{b+1}', spatial_downsample
            )
            self.classifier.add_module(
                f'TemporalBlock{b+1}', spatial_downsample
            )
        final_block = _CNetFinalBlock(
            feature_config=cnet_output_feature_config
            activ_func=activ_func_cnet,
            conv_size=conv_kernel_size_cnet,
            dropout_rate=dropout_rate_cnet,
            norm_func=norm_func_cnet
        )
        self.classifier.add_module('FinalConvBlock', final_block)

        
    ###
    def forward(self, x):
        def _squeeze_spatial(x):
            while x.shape[-1] == 1:
                x = torch.squeeze(x, dim=-1)
            return x

        def _restore_dims(x, nD):
            while len(x.shape) != nD:
                x = torch.unsqueeze(x, dim=-1)
            return x

        # Storage
        enc = [None] * (self.n_blocks) # encoding level outputs
        dec = [None] * (self.n_blocks) # decoding level outputs
        cls = [None] * (self.n_blocks) # inputs to classifer
        siz = [None] * (self.n_blocks) # maxunpool output size
                
        # U-Net (encoding)
        for b in range(self.n_blocks):
            out = self.unet_encoder.__getattr__(f'ConvBlock{b+1}')(x)
            x = enc[b] = out[-1]
            siz[b] = enc[b].shape

            if b != self.n_blocks - 1:
                x = self.unet_encoder.__getattr__(f'Downsample{b+1}')(x)

        # U-Net (decoding)
        for b in reversed(range(self.n_blocks)):
            if b != self.n_blocks - 1:
                x = self.unet_decoder.__getattr__(f'Upsample{b+1}')(x, siz[b])
                out = self.unet_decoder.__getattr__(f'ConvBlock{b+1}')(
                    torch.cat([x, enc[b]], dim=1)
                )
            x = dec[b] = out[-1]
            cls[b] = out[-(self.L)]
            
        # Classifier
        """
        for b in range(self.n_blocks):
            sampling_kernel = self.classifier
            while cls[b].shape[-self.X:] != (1,) * self.X:
                cls[b] = self.classifier.__getattr__(f'Downsample{b+1}')(
                    cls[b]
                )
        """
        c = self.classifier.__getattr__('FinalConvBlock')(cls)
        breapkoint()
        return x, c



#------------------------------------------------------------------------------
#                           Up/downSampling functions
#------------------------------------------------------------------------------

class _DownSample(nn.Module):
    def __init__(self,
                 n_features:int,
                 sampling_factor:list[int],
                 X:int=3,
                 T:int=2,
                 temporal:bool=True
    ):
        super(_DownSample, self).__init__()
        self.T = T
        self.temporal = temporal
        
        
        func = Conv4d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=sampling_factor,
            stride=sampling_factor,
            X=X,
            T=T,
        ) if self.temporal else eval(f'nn.Conv{X}d')(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=sampling_factor,
            stride=sampling_factor,
        )
        self.add_module('downsample', func)
        
        
    def forward(self, x, db=False):
        x = (self.downconv(x) if self.temporal
             else torch.stack(
                     [self.downconv(xt) for xt in x.movedim(self.T, 0)], dim=0
             ).movedim(0, self.T)
        )
        return x



class _UpSample(nn.Module):
    def __init__(self,
                 n_features:int,
                 sampling_factor:list[int],
                 X:int,
                 T:int=2,
                 temporal:bool=True,
    ):
        super(_UpSample, self).__init__()
        self.T = T
        self.temporal = temporal

        func = Conv4d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=sampling_factor,
            padding=1,
            X=X,
            T=T,
        ) if self.temporal else eval(f'nn.Conv{X}d')(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=sampling_factor,
            padding=1,
        )
        self.add_module('upsample', func)


    def _repeat_interleave(self, x):
        for d in range(self.T if self.temporal else self.T+1, len(x.shape)):
            x = x.repeat_interleave(self.factor, dim=d)
        return x
    

    def forward(self, x, out_shape):
        x = self._repeat_interleave(x)
        x = (
            self.upsample(x) if self.temporal
            else torch.stack(
                    [self.upsample(xt)for xt in x.movedim(self.T, 0)],
                    dim=self.T
            )
        )
        return x

    
class _MaxPool(nn.Module):
    def __init__(self,
                 n_features:int,
                 sampling_factor:list[int],
                 X:int,
                 T:int=2,
                 temporal:bool=True,
    ):
        super(_MaxPool, self).__init__()
        self.T = T
        self.temporal = temporal
        
        spatial_func = eval(f'nn.MaxPool{X}d')(
            kernel_size=sampling_factor[1:],
            stride=spatial_factor,
            return_indices=True
        )
        self.add_module('pool_spatial', spatial_func)

        temporal_func = nn.MaxPool2d(
            kernel_size=(sampling_factor[0], 1),
            stride=(temporal_factor, 1),
            return_indices=True
        )
        self.add_module(
            'pool_temporal', temporal_func if self.temporal else None
        )
        

    def forward(self, x):
        nB, nC, nT = x.shape[:self.T+1]
        nvox = np.prod(x_pool_spatial.shape[self.T+1:])

        # spatial pooling
        x_pool_spatial, inds_spatial = [
            self.pool_spatial(xt) for xt in x.movedim(self.T, 0)
        ]
        x_pool_spatial = torch.stack(x_pool_spatial, dim=self.T)
        
        # temporal pooling
        if self.temporal:
            x_pool, inds_temporal = self.pool_temporal(
                x_pool_spatial.view(nB, nC, nT, nvox)
            )
            x_pool = x_pool.view(
                nB, nC, -1, *x_pool_spatial.shape[self.T+1:])
        else:
            x_pool = x_pool_spatial
            inds_temporal = None
            
        return x_pool, inds_spatial, inds_temporal
        

    
class _MaxUnpool(nn.Module):
    def __init__(self,
                 X:int,
                 T:int=2,
                 spatial_kernel:int=2,
                 spatial_stride:int=2,
                 temporal_kernel:int=2,
                 temporal_stride:int=2,
                 temporal:bool=True,
    ):
        super(_MaxUnpool, self).__init__()    
        
        self.X = X
        self.T = T
        self.temporal = temporal

        spatial_func = eval(f'nn.MaxUnPool{X}d')(
            kernel_size=sampling_factor[1:],
            stride=spatial_factor,
        )
        self.add_module('unpool_spatial', spatial_func)

        temporal_func = nn.MaxUnPool2d(
            kernel_size=(sampling_factor[0], 1),
            stride=(temporal_factor, 1),
        )
        self.add_module(
            'unpool_temporal', temporal_func if self.temporal else None
        )
                

    def forward(self, x, inds_spatial, inds_temporal, siz):
        nB, nC, nT = x.shape[:self.T+1]
        nvox = np.prod(x_pool_spatial.shape[self.T+1:])
        
        # temporal unpooling
        if self.temporal:
            x_flat = x.view(nB, nC, nT, -1)
            x_unpool_temporal = self.unpool_temporal(
                x_flat, inds_temporal, output_size=siz
            ).view(nB, nC, -1, *x.shape[self.T+1:])
        else:
            x_unpool_temporal = x
            
        # spatial unpooling
        x_unpool = torch.stack(
            [self.unpool_spatial(xt)
             for xt in x_unpool_temporal.movedim(self.T, 0)
            ], dim=self.T
        )        
        return x_unpool



#------------------------------------------------------------------------------
#                            Classifier functions
#------------------------------------------------------------------------------
    
class _CNetSpatialDownBlock(nn.ModuleDict):
    def __init__(self,
                 config:list[int],
                 norm_func:str=None,
                 activ_func:str=None,
                 dropout_rate:float=0.,
    ):
        super(_CNetBlock, self).__init__()
        n_layers = len(config) - 1
        
        for n in range(n_layers):
            layer =_CNetLayer(n_input_features=config[n],
                              n_output_features=config[n+1],
                              norm_func=norm_func,
                              activ_func=activ_func,
                              dropout_rate=dropout_rate,
            )
            self.add_module('ConvLayer%d' % (n + 1), layer)
            
            
    def forward(self, x):
        x_out = [None] * (len(self.items()) + 1)
        x_out[0] = x
        for n, [name, layer] in enumerate(self.items()):
            x_out[n+1] = layer(x_out[n])
        return x_out

    

class _CNetLayer(nn.Module):
    def __init__(self,
                 n_input_features:int,
                 n_output_features:int,
                 norm_func:str=None,
                 activ_func:str=None,
                 dropout_rate=0,
                 **kwargs
    ):
        super(_CNetLayer, self).__init__()

        activ = eval('nn.%s' % activ_func)() if activ_func is not None else nn.Identity()
        conv = nn.Linear(n_input_features,
                         n_output_features,
                         bias=False # if norm_func is not None else True
        )
        self.add_module('activ', activ if activ_func is not None else nn.Identity())
        self.add_module('conv', conv)


    def forward(self, x):
        x = torch.concatenate(x, dim=1)
        return self.conv(self.activ(x))


    
#------------------------------------------------------------------------------
#                               U-Net functions
#------------------------------------------------------------------------------

class _UNetBlock(nn.ModuleDict):
    def __init__(self,
                 feature_config:list[int],
                 activ_func:str=None,
                 conv_size:list[int]=3,
                 conv_shape:str='hypercube',
                 dropout_rate:float=0.,
                 norm_func:str=None,
                 residual:bool=False,
                 skip:bool=False,
                 temporal:bool=False,
                 X:int=3,
                 T:int=2
    ):
        super(_UNetBlock, self).__init__()
        self.use_residuals = residual

        # Parse args
        n_layers = len(feature_config) - 1
        layer_func = '_'.join(
            ['_UNetLayer', 'long' if temporal else 'standard']
        )
        self.use_residuals = residual
        
        for n in range(n_layers):
            growth = 1 + (skip and n == n_layers - 1)
            layer = eval(f'{layer_func}')(
                n_input_features=feature_config[n],
                n_output_features=growth*feature_config[n+1],
                activ_func=activ_func,
                conv_size=conv_size,
                conv_shape=conv_shape,
                dropout_rate=dropout_rate,
                norm_func=norm_func,
                X=X,
                T=T
            )
            self.add_module(f'ConvLayer{n+1}', layer)


    def forward(self, x, db=False):
        x_out = [None] * (len(self.items()) + 1)
        x_out[0] = x

        for n, [name, layer] in enumerate(self.items()):
            res = x_out[n]
            x_out[n+1] = layer(x_out[n], db=db)
            
            if self.residuals and name[-1]!='1':
                x_out[n+1] += x_out[n]

        return x_out



class _UNetBlockTranspose(nn.ModuleDict):
    def __init__(self,
                 feature_config:list[int],
                 activ_func:str=None,
                 conv_size:list[int]=3,
                 conv_shape:str='hypercube',
                 dropout_rate:float=0.,
                 norm_func:str=None,
                 residual:bool=False,
                 skip:bool=False,
                 temporal:bool=False,
                 X:int=3,
                 T:int=2
    ):
        super(_UNetBlockTranspose, self).__init__()

        # Parse args
        n_layers = len(feature_config) - 1
        layer_func = '_'.join(
            ['_UNetLayerTranspose', 'long' if temporal else 'standard']
        )        

        # Add each layer
        for n in range(n_layers):
            growth = 1 + (skip and n == 0)
            layer = eval(f'{layer_func}')(
                n_input_features=growth*feature_config[n],
                n_output_features=feature_config[n+1],
                activ_func=activ_func,
                conv_size=conv_size,
                conv_shape=conv_shape,
                dropout_rate=dropout_rate,
                norm_func=norm_func,
                X=X,
                T=T
            )
            self.add_module(f'ConvLayer{n+1}', layer)


    def forward(self, x):
        x_out = [None] * (len(self.items()) + 1)
        x_out[0] = x

        for n, [name, layer] in enumerate(self.items()):
            x_out[n+1] = layer(x_out[n])

            if self.residuals and name[-1]!='1':
                x_out[n+1] += x_out[n]
            
        return x_out


    
class _UNetLayer_standard(nn.Module):
    def __init__(self,
                 n_input_features:int,
                 n_output_features:int,
                 activ_func:str=None,
                 conv_size:list[int]=3,
                 conv_shape:str=None,
                 dropout_rate:float=0.,
                 norm_func:str=None,
                 X:int=3,
                 T:int=2
    ):
        super(_UNetLayer_standard, self).__init__()

        # Parse args
        conv_kernel_size = (
            [conv_size] * X if isinstance(conv_size, int) else conv_size
        )
        conv_padding_size = [
            ((cs - 1) // 2) if cs % 2 == 1 else (cs // 2)
            for cs in conv_kernel_size
        ]
        self.T = T

        # Define functions
        activ = (
            activ_func() if activ_func is not None
            else nn.Identity()
        )
        conv = eval('nn.Conv{X}d')(
            in_channels=n_input_features,
            out_channels=n_output_features,
            kernel_size=conv_kernel_size,
            padding=conv_padding_size,
            bias=False if norm_func is not None else True
        )
        drop = (
            eval('nn.Dropout{X}d')(p=dropout_rate) if dropout_rate > 0.
            else nn.Identity()
        )
        norm = (
            norm_func(n_input_features) if norm_func is not None
            else nn.Identity()
        )

        # Add to class
        self.add_module('norm', norm)
        self.add_module('activ', activ)
        self.add_module('conv', conv)
        self.add_module('drop', drop)

        
    def forward(self, x):
        x = torch.stack(
            [self.drop(self.conv(self.activ(self.drop(xt))))
             for xt in x.movedim(self.T, 0)
            ], dim=self.T
        )
        return x


    
class _UNetLayerTranspose_standard(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 T:int=2,
                 norm_func:str=None,
                 activ_func:str=None,
                 dropout_rate=0,
                 **kwargs
    ):
        super(_UNetLayerTranspose_standard, self).__init__()

        # Parse args
        conv_kernel_size = (
            [conv_size] * X if isinstance(conv_size, int) else conv_size
        )
        conv_padding_size = [
            ((cs - 1) // 2) if cs % 2 == 1 else (cs // 2)
            for cs in conv_kernel_size
        ]
        self.T = T

        # Define functions
        activ = (
            activ_func() if activ_func is not None
            else nn.Identity()
        )
        conv = eval('nn.ConvTranspose{X}d')(
            in_channels=n_input_features,
            out_channels=n_output_features,
            kernel_size=conv_kernel_size,
            padding=conv_padding_size,
            bias=False if norm_func is not None else True
        )
        drop = (
            eval('nn.Dropout{X}d')(p=dropout_rate) if dropout_rate > 0.
            else nn.Identity()
        )
        norm = (
            norm_func(n_input_features) if norm_func is not None
            else nn.Identity()
        )

        # Add to class
        self.add_module('drop', drop)
        self.add_module('norm', norm)
        self.add_module('activ', activ)
        self.add_module('conv', conv)


    def forward(self, x):
        x = torch.stack(
            [self.conv(self.activ(self.norm(self.drop(xt))))
             for xt in x.movedim(self.T, 0)
            ], dim=self.T
        )
        return x



class _UNetLayer_long(nn.Module):
    def __init__(self,
                 n_input_features:int,
                 n_output_features:int,
                 activ_func:str=None,
                 conv_size:list[int]=3,
                 conv_shape:str='hypercube',
                 dropout_rate:float=0.,
                 norm_func:str=None,
                 X:int=3,
                 T:int=2
    ):
        super(_UNetLayer_long, self).__init__()

        # Parse args
        conv_kernel_size = (
            [conv_size] * X if isinstance(conv_size, int) else conv_size
        )
        conv_padding_size = [
            ((cs - 1) // 2) if cs % 2 == 1 else (cs // 2)
            for cs in conv_kernel_size
        ]
        self.T = T
        
        # Define functions
        activ = (
            activ_func() if activ_func is not None
            else nn.Identity()
        )
        conv = Conv4d(
            in_channels=n_input_features,
            out_channels=n_output_features,
            kernel_size=conv_kernel_size,
            kernel_shape=conv_shape,
            padding=conv_padding_size,
            bias=False if norm_func is not None else True
        )
        drop = (
            eval('nn.Dropout{X}d')(p=dropout_rate) if dropout_rate > 0.
            else nn.Identity()
        )
        norm = (
            norm_func(n_input_features) if norm_func is not None
            else nn.Identity()
        )

        # Add to class
        self.add_module('drop', drop)
        self.add_module('norm', norm)
        self.add_module('activ', activ)
        self.add_module('conv', conv)


    def forward(self, x, db=False):
        x = torch.stack(
            [self.drop(xt) for xt in torch.stack(
                [self.conv(self.activ(self.norm(self.drop(xt))))
                 for xt in x.movedim(self.T, 0)
                ], dim=self.T).movedim(self.T, 0)
            ], dim=self.T
        )
        return x



class _UNetLayerTranspose_long(nn.Module):
    def __init__(self,
                 n_input_features:int,
                 n_output_features:int,
                 activ_func:str=None,
                 conv_size:list[int]=3,
                 conv_shape:str='hypercube',
                 dropout_rate:float=0.,
                 norm_func:str=None,
                 X:int=3,
                 T:int=2
    ):
        super(_UNetLayerTranspose_long, self).__init__()

        # Parse args
        conv_kernel_size = (
            [conv_size] * X if isinstance(conv_size, int) else conv_size
        )
        conv_padding_size = [
            ((cs - 1) // 2) if cs % 2 == 1 else (cs // 2)
            for cs in conv_kernel_size
        ]
        self.T = T

        # Define functions
        activ = (
            activ_func() if activ_func is not None
            else nn.Identity()
        )
        conv = ConvTranspose4d(
            in_channels=n_input_features,
            out_channels=n_output_features,
            kernel_size=conv_kernel_size,
            kernel_shape=conv_shape,
            padding=conv_padding_size,
            bias=False if norm_func is not None else True
        )
        drop = (
            eval('nn.Dropout{X}d')(p=dropout_rate) if dropout_rate > 0.
            else nn.Identity()
        )
        norm = (
            norm_func(n_input_features) if norm_func is not None
            else nn.Identity()
        )

        # Add to class
        self.add_module('drop', drop)
        self.add_module('norm', norm)
        self.add_module('activ', activ)
        self.add_module('conv', conv)
        

    def forward(self, x):
        x = self.conv(
            torch.stack(
                [self.activ(self.norm(self.drop(xt)))
                 for xt in x.movedim(self.T, 0)
                ], dim=self.T
            )
        )
        return x

