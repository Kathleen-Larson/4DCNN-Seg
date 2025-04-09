import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from conv4d import Conv4d, ConvTranspose4d
import utils


#------------------------------------------------------------------------------

class UClassNetXD_Long(nn.Module):
    """
    Joint unet+classifier change detection network.
    
    Required inputs:
    - in_channels:        no. input channels (most likely image modalities)
    - out_channels_unet:  no. output channels for unet (label classes)
    - out_channels_cnet:  no. output channels for cnet (total disease classes)

    Optional args (general):
    - feature_ratio:        factor to increase features at each level
    - n_levels:             number of unet/cnet levels
    - n_starting_features:  number of features at highest spatial res.
    - transfer_index:       index of unet decoder output for cnet input 
    - use_only_cnet:        flag to use only cnet component of UClassNetXD
    - use_only_unet:        flag to use only unet component of UClassNetXD
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
                 conv_kernel_size_cnet:list[int]=3,
                 conv_kernel_size_unet:list[int]=3,
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
                 sample_func_down_cnet:str='_DownConv',
                 sample_func_down_unet:str='_DownConv',
                 sample_func_up_unet:str='_UpConv',
                 sample_kernel_size_cnet:list[int]=2,
                 sample_kernel_size_unet:list[int]=2,
                 transfer_index:int=2,
                 use_only_cnet:bool=False,
                 use_only_unet:bool=False,
                 use_residuals:bool=False,
                 use_skips:bool=True,
                 use_temporal:str=True,
                 use_unet_only:bool=False,
                 T:int=2,
                 X:int=3,
    ):
        super(UClassNetXD_Long, self).__init__()
        
        # Parse class attributes
        self.do_cnet = True if not use_only_unet else False
        self.do_unet = True if not use_only_cnet else False
        self.use_residuals = use_residuals
        self.use_skips = use_skips
        self.use_temporal = use_temporal
        self.L = transfer_index
        self.T = T
        self.X = X

        feature_config = [
            int(n_starting_features * (feature_ratio ** n))
            for n in range(n_levels)
        ]
        self.n_levels = n_levels
        
        # Functions (activation, normalization, sampling)
        activ_func_cnet = _parse_arg_as_function(self, activ_func_cnet)
        activ_func_unet = _parse_arg_as_function(self, activ_func_unet)

        norm_func_cnet = _parse_arg_as_function(self, norm_func_cnet)
        norm_func_unet = _parse_arg_as_function(self, norm_func_unet)

        sample_func_down_cnet = _parse_arg_as_function(
            self, sample_func_down_cnet)
        sample_func_down_unet = _parse_arg_as_function(
            self, sample_func_down_unet)
        sample_func_up_unet = _parse_arg_as_function(
            self, sample_func_up_unet)

        sample_kernel_size_cnet = _parse_arg_as_list(
            self, sample_kernel_size_cnet, int)
        sample_kernel_size_unet = _parse_arg_as_list(
            self, sample_kernel_size_unet, int)
        
        # Convolution kernels
        conv_shapes = ['hypercube', 'hypercross']
        conv_shape_cnet = _valid_configured_input(
            self, conv_shape_cnet, conv_shapes)
        conv_shape_unet = _valid_configured_input(
            self, conv_shape_unet, conv_shapes)
        
        conv_kernel_size_cnet = _parse_arg_as_list(
            self, conv_kernel_size_cnet, int)
        conv_kernel_size_unet = _parse_arg_as_list(
            self, conv_kernel_size_unet, int)
           
        # Build unet
        if self.do_unet:
            # Encoding blocks
            self.encoder = nn.Module()
            unet_enc_config = [in_channels] + feature_config

            for n in range(self.n_levels):
                block_config = (
                    [unet_enc_config[n]]
                    + [unet_enc_config[n+1]] * n_convs_per_block_unet
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
                self.encoder.add_module(f'ConvBlock{n+1}', convblock)
                self.encoder.add_module(f'Downsample{n+1}', downsample)

            # Decoding blocks
            self.decoder = nn.Module()
            unet_dec_config = [out_channels_unet] + feature_config            

            for n in reversed(range(self.n_levels)):
                block_config = (
                    [unet_dec_config[n+1]] * n_convs_per_block_unet
                    + [unet_dec_config[n]]
                )
                upsample = sample_func_up_unet(
                    n_features=block_config[0],
                    sampling_factor=sample_kernel_size_unet,
                    temporal=False,
                    X=self.X,
                    T=self.T,
                )
                convblock = _UNetBlock(
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
                self.decoder.add_module(f'Upsample{n+1}', upsample)
                self.decoder.add_module(f'ConvBlock{n+1}', convblock)
                
                
        # Build classifier arm
        if self.do_cnet:
            self.classifier = nn.Module()
            cnet_input_feature_config = unet_dec_config[1:]

            # Downsampling
            for b in range(self.n_levels):
                spatial_downsample = _CNetSpatialDownBlock(
                    n_features=cnet_input_feature_config[b],
                    sampling_func=sample_func_down_cnet,
                    sampling_factor=sample_kernel_size_unet,
                    X=self.X,
                    T=self.T
                )
                temporal_downsample = _CNetTemporalDownBlock(
                    n_features=cnet_input_feature_config[b],
                    sampling_factor=sample_kernel_size_cnet[0],
                    T=self.T,
                )
                self.classifier.add_module(
                    f'Downsample{b+1}', nn.Sequential(
                        spatial_downsample, temporal_downsample
                    )
                )

            # Final linear functions
            n_cnet_features = torch.tensor(
                cnet_input_feature_config).sum().item()
            cnet_output_feature_config = (
                [n_cnet_features] * n_convs_per_block_cnet
                + [out_channels_cnet]
            )
            final_block = _CNetFinalBlock(
                feature_config=cnet_output_feature_config,
                activ_func=activ_func_cnet,
                conv_size=conv_kernel_size_cnet[0],
            )
            self.classifier.add_module('FinalConvBlock', final_block)


    # Forward call
    def forward(self, x):
        enc = [None] * (self.n_levels) # encoding level outputs
        dec = [None] * (self.n_levels) # decoding level outputs
        cls = [None] * (self.n_levels) # inputs to classifer
        siz = [None] * (self.n_levels) # maxunpool output size

        # U-Net
        if self.do_unet:
            for n in range(self.n_levels):
                out = self.encoder.__getattr__(f'ConvBlock{n+1}')(x)
                x = enc[n] = out[-1]
                siz[n] = enc[n].shape
                
                if n != self.n_levels - 1:
                    x = self.encoder.__getattr__(f'Downsample{n+1}')(x)
            
            for n in reversed(range(self.n_levels)):
                if n != self.n_levels - 1:
                    x = self.decoder.__getattr__(f'Upsample{n+1}')(x, siz[n])
                out = self.decoder.__getattr__(f'ConvBlock{n+1}')(
                    torch.cat([enc[n], x], dim=1)
                )
                x = dec[n] = out[-1]
                cls[n] = out[-(self.L)]
        else:
            # this is where you figure out how to just do the classifier stuff
            breakpoint()

                
        # Classifier
        if self.do_cnet:
            for n in range(self.n_levels):
                cls[n] = self.classifier.__getattr__(f'Downsample{n+1}')(
                    cls[n]
                )                
            c = self.classifier.FinalConvBlock(cls)
    

        return x, c


#------------------------------------------------------------------------------

#class UNetXD_Longitudinal(nn.Module):
    """
    Main class for the 4D unet segmentation network. Input arguments will have
    the same name convention as UClassNetXD (e.g. <param>_unet) even though 
    this class not not directly involve the classifier.

    Required inputs:
    - in_channels:        no. input channels (most likely image modalities)
    - out_channels_unet:  no. output channels for unet (label classes)

    Optional args (general):
    - feature_ratio:        factor to increase features at each level
    - n_levels:             number of cnet levels
    - n_starting_features:  number of features at highest spatial res.
    - T:                    index of temporal dimension in input image
    - X:                    number of spatial dimensions in input image

    Optional args (unet):
    - activ_func_unet           activation function for conv layers
    - conv_kernel_size_unet:    size of 4D conv kernel
    - conv_shape_unet:          shape of 4D conv kernel (hypercube/hypercross)
    - drop_rate_unet:           dropout rate in conv layers (0 = no dropout)
    - feature_ratio:            factor to increase features at each level
    - n_convs_per_block_unet:   number of conv layers per unet level
    - norm_func_unet:           normalization function for conv layers
    - sample_down_func_unet:    downsampling function for enc. arm
    - sample_up_func_unet:      upsampling function for dec. arm
    - sample_kernel_size_unet:  size of sampling pool/conv kernel
    - use_residuals:            flag for residual conns. in enc. arm
    - use skips:                flag for skip conns. between enc./dec. arms 
    """
    
    


#------------------------------------------------------------------------------
#                       Up/down sampling functions
#------------------------------------------------------------------------------

class _DownConv(nn.Module):
    def __init__(self,
                 n_features:int,
                 sampling_factor:list[int],
                 X:int=3,
                 T:int=2,
                 temporal:bool=True
    ):
        super(_DownConv, self).__init__()

        # Parse class args
        self.T = T
        self.temporal = temporal

        # Define sampling function
        self.downsample = Conv4d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=sampling_factor,
            stride=sampling_factor,
            bias=False,
            X=X,
            T=T,
        ) if self.temporal else eval(f'nn.Conv{X}d')(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=sampling_factor[-X:],
            stride=sampling_factor[-X:],
            bias=False
        )
        
        
    def forward(self, x, db=False):
        x = (self.downsample(x) if self.temporal
             else torch.stack(
                     [self.downsample(xt)
                      for xt in x.movedim(self.T, 0)
                     ], dim=self.T
             )
        )
        return x



class _UpConv(nn.Module):
    def __init__(self,
                 n_features:int,
                 sampling_factor:list[int],
                 X:int,
                 T:int=2,
                 temporal:bool=True,
    ):
        super(_UpConv, self).__init__()

        # Parse class args
        self.T = T
        self.sampling_factor = sampling_factor
        self.temporal = temporal

        # Define sampling function
        self.upsample = Conv4dTranspose(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=sampling_factor,
            bias=False,
            X=X,
            T=T,
        ) if self.temporal else eval(f'nn.ConvTranspose{X}d')(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=sampling_factor[-X:],
            stride=sampling_factor[-X:],
            bias=False
        )
        

    def forward(self, x, out_sz):
        x = (
            self.upsample(x) if self.temporal
            else torch.stack(
                    [self.upsample(xt) for xt in x.movedim(self.T, 0)],
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

        # Parse class args
        self.T = T
        self.temporal = temporal

        # Define pooling functions
        self.pool_spatial = eval(f'nn.MaxPool{X}d')(
            kernel_size=sampling_factor[1:],
            stride=spatial_factor,
            return_indices=True
        )
        self.pool_temporal = nn.MaxPool2d(
            kernel_size=(sampling_factor[0], 1),
            stride=(temporal_factor, 1),
            return_indices=True
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

        # Parse args
        self.X = X
        self.T = T
        self.temporal = temporal

        # Define pooling function
        self.unpool_spatial = eval(f'nn.MaxUnPool{X}d')(
            kernel_size=sampling_factor[-X:],
            stride=spatial_factor[-X:],
        )

        self.unpool_temporal = nn.MaxUnPool2d(
            kernel_size=(sampling_factor[0], 1),
            stride=(temporal_factor, 1),
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

class _CNetSpatialDownBlock(nn.Module):
    """
    Takes an input from the unet decoder and downsamples all spatial dimensions
    to a single element (e.g. [B, C, T, H, W, D]-->[B, C, T, 1, 1, 1])
    """
    def __init__(self,
                 n_features:int,
                 sampling_func,
                 sampling_factor:int=2,
                 X:int=3,
                 T:int=2,
    ):
        super(_CNetSpatialDownBlock, self).__init__()
        self.X = X
        self.T = T

        self.downsample = sampling_func(
            n_features=n_features,
            sampling_factor=sampling_factor,
            temporal=False,
            X=X,
            T=T
        )


    def forward(self, x):
        """
        TO-DO: might need to handle the case when x.shape is to conv but not
        quite [1, 1, 1]
        """
        while torch.where(torch.tensor(x.shape[-self.X:]) > 1)[0].numel() > 1:
            x = self.downsample(x)
            
        for _ in range(self.X):
            x = x.squeeze(-1)

        return x



class _CNetTemporalDownBlock(nn.Module):
    """
    Takes a fully spatially downsampled input and downsamples the temporal
    dimension (e.g. [B, C, T]-->[B, C, 1])
    """
    def __init__(self,
                 n_features:int,
                 sampling_factor:int=2,
                 T:int=2,
    ):
        super(_CNetTemporalDownBlock, self).__init__()
        self.T = T

        self.downsample = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=sampling_factor,
            stride=sampling_factor,
            bias=False
        )


    def forward(self, x):
        while x.shape[self.T] > 1:
            x = self.downsample(x)
        return x.squeeze(-1)



class _CNetFinalBlock(nn.ModuleDict):
    """
    Takes a fully downsampled input and produces the final classifier output
    """
    def __init__(self,
                 feature_config:list[int],
                 activ_func=None,
                 conv_size:int=3,
    ):
        super(_CNetFinalBlock, self).__init__()
        self.n_layers = len(feature_config) - 1
        padding_size = (
            (conv_size - 1) // 2 if conv_size % 2 == 1 else conv_size // 2
        )

        # Add intermediate conv layers
        for n in range(self.n_layers - 1):
            func = nn.Linear(
                in_features=feature_config[n],
                out_features=feature_config[n+1],
                bias=True
            )
            activ = (
                activ_func() if activ_func is not None
                else nn.Identity()
            )
            self.add_module(f'Layer{n+1}', nn.Sequential(func, activ))
            
        # Add last linear function
        self.linear = nn.Linear(
            in_features=feature_config[-2],
            out_features=feature_config[-1],
        )
                    

    def forward(self, x):
        x = torch.cat(x, dim=1)
        for n in range(self.n_layers - 1):
            x = self.__getattr__(f'Layer{n+1}')(x)
        x = self.linear(x)            
        return x



#------------------------------------------------------------------------------
#                     Unet block functions
#------------------------------------------------------------------------------

class _UNetBlock(nn.ModuleDict):
    def __init__(self,
                 feature_config:list[int],
                 activ_func=None,
                 conv_size:list[int]=3,
                 conv_shape:str='hypercube',
                 dropout_rate:float=0.,
                 norm_func=None,
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
        self.use_residuals = residual
        
        for n in range(n_layers):
            growth = 1 + (skip and n == 0)
            layer = _UNetLayer(
                n_input_features=growth*feature_config[n],
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


    def forward(self, x, db=False):
        x_out = [None] * (len(self.items()) + 1)
        x_out[0] = x

        for n, [name, layer] in enumerate(self.items()):
            res = x_out[n]
            x_out[n+1] = layer(x_out[n])
            
            if self.use_residuals and name[-1]!='1':
                x_out[n+1] += x_out[n]

        return x_out

    
    
class _UNetLayer(nn.Module):
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
        super(_UNetLayer, self).__init__()

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
        
        # Define functions
        self.conv = Conv4d(
            in_channels=n_input_features,
            out_channels=n_output_features,
            kernel_size=conv_kernel_size,
            kernel_shape=conv_shape,
            padding=conv_padding_size,
            bias=False if norm_func is not None else True
	) if self.temporal else eval(f'nn.Conv{X}d')(
            in_channels=n_input_features,
            out_channels=n_output_features,
            kernel_size=conv_kernel_size[-X:],
            padding=conv_padding_size[-X:],
            bias=False if norm_func is not None else True
        )
        activ = (
            activ_func() if activ_func is not None
            else nn.Identity()
        )
        drop = (
            eval(f'nn.Dropout{X}d')(p=dropout_rate) if dropout_rate > 0.
            else nn.Identity()
        )
        norm = (
            norm_func(n_output_features) if norm_func is not None
            else nn.Identity()
        )
        self.layer_funcs = nn.Sequential(norm, activ, drop)

        
    def forward(self, x):
        x = torch.stack(
            [self.layer_funcs(xt) for xt in self.conv(x).movedim(self.T, 0)]
            if self.temporal else
            [self.layer_funcs(self.conv(xt)) for xt in x.movedim(self.T, 0)]
            , dim=self.T
        )
        return x


#------------------------------------------------------------------------------
#                              Utility functions
#------------------------------------------------------------------------------

def _print_arg_error(instr:str, cls=None):
    func = 'unet4d_classifier.py' if cls is None else cls.__class__.__name__
    if cls is None:
        utils.fatal(f'Error in {func}.__init__: {instr}')


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
            _print_arg_error(
                f'{varname} must be {dtype} or list/tuple of {dtype} of '
                f'len=={cls.X+1} (input was {arg})'
            )
    return arg

        
def _parse_arg_as_function(cls, arg):
    try:
        arg = eval(f'{arg}')
    except:
        try:
            arg = eval(f'{arg}{cls.X}d')
        except:
            varname = f'{arg=}'.split('=')[0]
            _print_arg_error(
                f'{arg} (or {arg}{cls.X}d) is not a valid input for {varname} '
                f'(not a valid callable function)'
            )
    return arg

    
def _valid_configured_input(cls, arg, valid_list):
    if arg not in valid_list:
        varname = f'{arg=}'.split('=')[0]
        _print_arg_error(
            f'{arg} not a valid input for {varname} (must be in '
            f'{valid_list})'
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
    
    
