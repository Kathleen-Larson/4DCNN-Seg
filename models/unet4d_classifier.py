import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from .conv4d import Conv4d, ConvTranspose4d


###################################################################################################

class UNetXD_Longitudinal(nn.Module):
    def __init__(self,
                 in_channels:int,                          # n. of in channels
                 out_channels_unet:int,                    # n. of out channels (unet)
                 out_channels_cnet:int,                    # n. of out channels (classnet)
                 feature_config:list[int]=(24,48,96),      # n. of feature at each level
                 n_levels:int=3,                           # n. of levels
                 n_starting_features:int=24,               # n. of features for 1st conv layer
                 n_layers_unet:int=2,                      # n. of layers per block (unet)
                 n_layers_cnet:int=2,                      # n. of layers per block (classnet)
                 kernel_type_unet:str='hypercube',         # 4D kernel shape (unet)
                 kernel_type_cnet:str='hypercube',         # 4D kernel shape (classnet)
                 kernel_size_unet:[int, tuple]=(3,3,3,3),  # size of conv kernel (unet) 
                 kernel_size_cnet:[int, tuple]=(3,3,3,3),  # size of conv kernel (classnet)
                 norm_type_unet:str='Instance',            # norm fn. type (unet)
                 norm_type_cnet:str='Instance',            # norm fn. type (classnet)
                 activ_type_unet:str='ReLU',               # activation fn. type (unet)
                 activ_type_cnet:str='ReLU',               # activation fn. type (classnet)
                 pool_type_unet:str='conv',                # fn. for up/down sampling (unet)
                 pool_type_cnet:str='conv',                # fn. for up/down sampling (classnet)
                 temporal:str=True,                        # flag to turn on/off 4D convs.
                 skip:bool=True,                           # flag to turn on/off skip conns.
                 X:int=3,                                  # n. of spatial dims.
                 T:int=2,                                  # index of temporal dim.
                 L:int=1,                                  # index of dec. layers to give classnet 
                 **kwargs,
    ):
        super(UNetXD_Longitudinal, self).__init__()

        ## Parse args
        self.in_channels = in_channels
        self.temporal = temporal
        self.skip = skip
        self.X = X
        self.T = T
        self.L = L
        
        self.feature_config = [n_starting_features * (2**n) for n in range(n_levels)]
        self.n_blocks = len(self.feature_config)
                
        self.out_channels_unet = out_channels_unet
        self.out_channels_unet = out_channels_unet
        self.n_layers_unet = n_layers_unet
        self.n_layers_cnet = n_layers_cnet
        self.kernel_type_unet = kernel_type_unet
        self.kernel_type_cnet = kernel_type_cnet
        self.norm_type_unet = norm_type_unet
        self.norm_type_cnet = norm_type_cnet
        self.pool_type_unet = pool_type_unet
        self.pool_type_cnet = pool_type_cnet
        
        self.kernel_size_unet = [kernel_size_unet] * (X+1) \
            if isinstance(kernel_size_unet, int) \
               else kernel_size_unet
        assert len(kernel_size_unet) == X + 1, \
            "input kernel_size_unet must have length equal to spatial dims + 1"

        self.kernel_size_cnet = [kernel_size_cnet] * (X+1) \
            if isinstance(kernel_size_cnet, int) \
               else kernel_size_cnet
        assert len(kernel_size_cnet) == X + 1, \
            "input kernel_size_cnet must have length equal to spatial dims + 1"
        
        self.activ_type_unet = activ_type_unet \
            if callable(getattr(nn, activ_type_unet)) \
               else Exception("Invalid activ_type_unet (not an attribute of torch.nn")

        self.activ_type_cnet = activ_type_cnet \
            if callable(getattr(nn, activ_type_cnet)) \
               else Exception("Invalid activ_type_cnet (not an attribute of torch.nn")

        
        ## UNet encoding arm:
        self.encoder = nn.Sequential()
        encoding_config = [self.in_channels] + self.feature_config
        
        for b in range(len(encoding_config)-1):
            block_config = [encoding_config[b]] \
                + [encoding_config[b+1]] * self.n_layers_unet

            # Conv layer
            block = _UNetBlock(config=block_config,
                              layer_type='long' if self.temporal else 'standard',
                              norm_type=self.norm_type_unet,
                              activ_type=self.activ_type_unet,
                              X=self.X, T=self.T
            )
            self.encoder.add_module('ConvBlock%d' % (b+1), block)
            
            # Downsampling/pooling
            if self.pool_type_unet == 'conv':
                downsample = _DownSample(n_input_features=block_config[-1],
                                         n_output_features=block_config[-1],
                                         factor=2, X=self.X, temporal=False
                )
            elif self.pool_type_unet == 'maxpool':
                downsample = _MaxPool(X=self.X, T=self.T, temporal=False,
                                      spatial_kernel=2, spatial_stride=2,
                )
            self.encoder.add_module('Downsample%d' % (b+1), downsample)
                    
                    
        ## UNet decoding arm:
        self.decoder = nn.Sequential()
        decoding_config = [self.out_channels_unet] + self.feature_config
        
        for b in reversed(range(len(decoding_config) - 1)):
            block_config = [decoding_config[b+1]] * self.n_layers_unet \
                + [decoding_config[b]]
            
            # Upsampling/unpooling
            if self.pool_type_unet == 'conv':
                upsample = _UpSample(n_input_features=block_config[0],
                                     n_output_features=block_config[0],
                                     factor=2, X=self.X, temporal=False
                )
                
            elif self.pool_type_unet == 'maxpool':
                upsample = _MaxUnpool(X=self.X, T=self.T, temporal=False,
                                      spatial_kernel=2, spatial_stride=2,
                )
            self.decoder.add_module('Upsample%d' % (b+1), upsample)
                
            # Conv layer
            block_config = [decoding_config[b+1]] * self.n_layers_unet \
                + [decoding_config[b]]
            block = _UNetBlockTranspose(config=block_config,
                                        layer_type='long' if self.temporal else 'standard',
                                        norm_type=self.norm_type_unet,
                                        activ_type=self.activ_type_unet,
                                        drop_rate=0, X=self.X, T=self.T
            )
            self.decoder.add_module('ConvBlock%d' % (b+1), block)
            
            
        ## Classifier arm
        self.classifier = nn.Sequential()
        n_cnet_features = torch.tensor(self.feature_config).sum().item()
        classifier_config = [n_cnet_features] * self.n_layers_cnet + [1]
        
        for b, n_in in enumerate(classifier_config):
            # Down sampling in spatial/temporal dimensions
            if self.pool_type_unet == 'conv':
                downsample = _DownSample(n_input_features=decoding_config[b+1],
                                         n_output_features=decoding_config[b+1],
                                         factor=2, X=self.X, temporal=False
                )
            elif self.pool_type_unet == 'maxpool':
                downsample = _MaxPool(X=self.X, T=self.T, temporal=False,
                                      spatial_kernel=2, spatial_stride=2,
                )
            self.classifier.add_module('Downsample%d' % (b+1), downsample)

        # Reduce feature dimensions
        block = _UNetBlock(config=classifier_config,
                           layer_type='long' if self.temporal else 'standard',
                           norm_type=self.norm_type_unet,
                           activ_type=self.activ_type_unet,
                           drop_rate=0,
                           X=self.X,
                           T=self.T
        )
        self.classifier.add_module('FinalConvBlock', block)
        
    

    def forward(self, x):
        enc = [None] * (self.n_blocks) # encoding level outputs
        dec = [None] * (self.n_blocks) # decoding level outputs
        cls = [None] * (self.n_blocks) # inputs to classifer
        siz = [None] * (self.n_blocks) # maxunpool output size
                
        # Encoding
        for b in range(self.n_blocks):
            out = self.encoding.__getattr__('ConvBlock%d' % (b+1))(x)
            x = enc[b] = out[-1]
            siz[b] = enc[b].shape

            if b != self.n_blocks - 1:
                x = self.encoding.__getattr__('Downsample%d' % (b+1))(x)

        # Decoding
        for b in reversed(range(self.n_blocks)):
            if b != self.n_blocks - 1:
                x = self.decoding.__getattr__('Upsample%d' % (b+1))(x, siz[b])

            out = self.decoding.__getattr__('ConvBlock%d' % (b+1))(torch.cat([x, enc[b]], 1))
            x = dec[b] = out[-1]
            cls[b] = out[-(self.L+1)]

        # Classifier
        #for b in range(self.n_blocks):            
            
        
        return x



class _DownSample(nn.Module):
    def __init__(self,
                 n_input_features:int,  # n. of input channels for convolution
                 n_output_features:int, # n. of output channels for convolution
                 factor:list[int],      # sampling factor
                 X:int,                 # n. of image dims.
                 T:int=2,               # index of temporal dim.
                 temporal:bool=True,    # flag to turn on/off downsampling in temporal dim.
    ):
        super(_DownSample, self).__init__()
        
        self.X = X
        self.T = T
        self.temporal = temporal

        downconv = Conv4d(in_channels=n_input_features,
                          out_channels=n_output_features,
                          kernel_size=factor,
                          stride=factor,
                          X=self.X,
                          T=self.T,
        ) if self.temporal else eval('nn.Conv%dd' % X)(in_channels=n_input_features,
                                                       out_channels=n_output_features,
                                                       kernel_size=factor,
                                                       stride=factor,
        )
        self.add_module('downconv', downconv)


    def forward(self, x):
        x = self.downconv(x) if self.temporal else \
            torch.stack([self.downconv(x[:,:,t,...]) for t in range(x.shape[self.T])], 2)
        return x



class _UpSample(nn.Module):
    def __init__(self,
                 n_input_features:int,  # n. of input channels for convolution
                 n_output_features:int, # n. of output channels for convolution
                 factor:list[int],      # sampling factor
                 X:int,                 # n. of image dims.
                 T:int=2,               # index of temporal dim.
                 temporal:bool=True,    # flag to turn on/off upsampling in temporal dim.
    ):
        super(_UpSample, self).__init__()

        self.X = X
        self.T = T
        self.factor = factor
        self.temporal = temporal

        upconv = Conv4d(in_channels=n_input_features,
                        out_channels=n_output_features,
                        kernel_size=factor+1,
                        padding=1,
                        X=self.X,
                        T=self.T,
        ) if self.temporal else eval('nn.Conv%dd' % X)(in_channels=n_input_features,
                                                       out_channels=n_output_features,
                                                       kernel_size=factor+1,
                                                       padding=1,
        )
        self.add_module('upconv', upconv)


    def _repeat_interleave(self, x):
        for i in range(self.T if self.temporal else self.T+1, len(x.shape)):
            x = x.repeat_interleave(self.factor, dim=i)
        return x
        

    def forward(self, x, out_shape):
        x = self._repeat_interleave(x)
        x = self.upconv(x) if self.temporal else \
            torch.stack([self.upconv(x[:,:,t,...]) for t in range(x.shape[self.T])], self.T)
        return x


    
class _MaxPool(nn.Module):
    def __init__(self,
                 X:int,
                 T:int=2,
                 spatial_kernel:int=2,
                 spatial_stride:int=2,
                 temporal_kernel:int=2,
                 temporal_stride:int=2,
                 temporal:bool=True,
    ):
        super(_MaxPool, self).__init__()
        
        self.X = X
        self.T = T
        self.temporal = temporal
        
        self.add_module('pool_spatial', eval('nn.MaxPool%dd' % self.X)(kernel_size=spatial_kernel,
                                                                       stride=spatial_stride,
                                                                       return_indices=True
        ))
        self.add_module('pool_temporal', nn.MaxPool1d(kernel_size=temporal_kernel,
                                                      stride=temporal_stride,
                                                      return_indices=True
        ) if self.temporal else None)
        

    def forward(self, x):
        # spatial pooling
        x_pool_spatial = [None] * x.shape[self.T]
        inds_spatial = [None] * x.shape[self.T]
        
        for t in range(len(x_pool_spatial)):
            x_pool_spatial[t], inds_spatial[t] = self.pool_spatial(x[:,:,t,...])
        x_pool_spatial = torch.stack(x_pool_spatial, dim=self.T)
            
    
        # temporal pooling
        if self.temporal:
            x_pool_spatial_1d = torch.reshape(x_pool_spatial, list(x_pool_spatial.shape[0:self.T+1]) + \
                                              list([np.prod(x_pool_spatial.shape[self.T+1:])]))
            x_pool = [None] * x_pool_spatial_1d.shape[-1]
            inds_temporal = [None] * x_pool_spatial_1d.shape[-1]
            
            for L in range(len(x_pool)):
                x_pool[L], inds_temporal[L] = self.pool_temporal(x_pool_spatial_1d[..., L])
            x_pool = torch.stack(x_pool, dim=-1)
            x_pool = torch.reshape(x_pool, list(x_pool.shape[0:self.T+1]) + list(x_pool_spatial.shape[self.T+1:]))
            
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
        
        self.add_module('unpool_spatial', eval('nn.MaxUnpool%dd' % X)(kernel_size=spatial_kernel,
                                                              stride=spatial_stride,
        ))
        self.add_module('unpool_temporal', nn.MaxUnpool1d(kernel_size=temporal_kernel,
                                                          stride=temporal_stride
        ) if self.temporal else None)
        

    def forward(self, x, inds_spatial, inds_temporal, siz):
        # temporal unpooling
        if self.temporal:
            x_1d = torch.reshape(x, list(x.shape[0:self.T+1]) + list([np.prod(x.shape[self.T+1:])]))
            x_unpool_temporal = [None] * x_1d.shape[-1]

            for L in range(x_1d.shape[-1]):
                x_unpool_temporal[L] = self.unpool_temporal(x_1d[...,L], inds_temporal[L], output_size=siz[0:self.T+1])
            x_unpool_temporal = torch.stack(x_unpool_temporal, dim=-1)
            x_unpool_temporal = torch.reshape(x_unpool_temporal, list(x_unpool_temporal.shape[0:self.T+1]) + \
                                              list(x.shape[self.T+1:]))
        else:
            x_unpool_temporal = x


        # spatial unpooling
        x_unpool = [None] * x_unpool_temporal.shape[self.T]

        for t in range(len(x_unpool)):
            x_unpool[t] = self.unpool_spatial(x_unpool_temporal[:,:,t,...], inds_spatial[t],
                                              list(siz[0:self.T]) + list(siz[self.T+1:]))
        x_unpool = torch.stack(x_unpool, dim=self.T)

        return x_unpool


    
class CNNLayer_standard(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 T:int=2,
                 norm_type:str=None,
                 activ_type:str=None,
                 drop_rate=0,
                 **kwargs
    ):
        super(CNNLayer_standard, self).__init__()
        self.T = T
        
        norm = eval('nn.%sNorm%dd' % (norm_type, X))(n_input_features)
        activ = eval('nn.%s' % activ_type)() if activ_type is not None else nn.Identity()
        conv = eval('nn.Conv%dd' % X)(n_input_features,
                                      n_output_features,
                                      kernel_size=3,
                                      padding=1,
                                      bias=False # if norm_type is not None else True
        )
        drop = eval('nn.Dropout%dd' % X)
        
        self.add_module('norm', norm if norm_type is not None else nn.Identity())
        self.add_module('activ', activ if activ_type is not None else nn.Identity())
        self.add_module('conv', conv)
        self.add_module('drop', drop if drop_rate > 0 else nn.Identity())

        
    def forward(self, x):
        x = torch.stack([self.norm(x[:,:,t,...]) for t in range(x.shape[self.T])], 2)
        x = self.activ(x)
        x = torch.stack([self.conv(x[:,:,t,...]) for t in range(x.shape[self.T])], 2)
        x = torch.stack([self.drop(x[:,:,t,...]) for t in range(x.shape[self.T])], 2)
        
        return x


    
class CNNLayerTranspose_standard(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 T:int=2,
                 norm_type:str=None,
                 activ_type:str=None,
                 drop_rate=0,
                 **kwargs
    ):
        super(CNNLayerTranspose_standard, self).__init__()
        self.T = T
        
        drop = eval('nn.Dropout%dd' % X)
        norm = eval('nn.%sNorm%dd' % (norm_type, X))(n_input_features)
        activ = eval('nn.%s' % activ_type)() if activ_type is not None else nn.Identity()
        conv = eval('nn.ConvTranspose%dd' % X)(n_input_features,
                                               n_output_features,
                                               kernel_size=3,
                                               padding=1,
                                               bias=False # if norm_type is not None else True
        )

        self.add_module('drop', drop if drop_rate > 0 else nn.Identity())
        self.add_module('norm', norm if norm_type else nn.Identity())
        self.add_module('activ', activ) # if activation_type is not None else nn.Identity())
        self.add_module('conv', conv)


    def forward(self, x):
        x = torch.stack([self.drop(x[:,:,t,...]) for t in range(x.shape[self.T])], 2)
        x = torch.stack([self.norm(x[:,:,t,...]) for t in range(x.shape[self.T])], 2)
        x = self.activ(x)
        x = torch.stack([self.conv(x[:,:,t,...]) for t in range(x.shape[self.T])], 2)
        
        return x




class CNNLayer_long(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 T:int=2,
                 kernel_size:[int, list]=3,
                 kernel_shape:str='hypercube',
                 norm_type:str=None,
                 activ_type:str=None,
                 drop_rate=0,
                 **kwargs
    ):
        super(CNNLayer_long, self).__init__()
        self.T = T
        
        norm = eval('nn.%sNorm%dd' % (norm_type, X))(n_input_features)
        activ = eval('nn.%s' % activ_type)() if activ_type is not None else nn.Identity()
        conv = Conv4d(in_channels=n_input_features,
                      out_channels=n_output_features,
                      kernel_size=kernel_size,
                      kernel_shape=kernel_shape,
                      padding=1,
                      X=X,
        )
        drop = eval('nn.Dropout%dd' % X)

        self.add_module('norm', norm if norm_type is not None else nn.Identity())
        self.add_module('activ', activ if activ_type is not None else nn.Identity())
        self.add_module('conv', conv)
        self.add_module('drop', drop if drop_rate > 0 else nn.Identity())


    def forward(self, x):
        x = torch.stack([self.norm(x[:,:,t,...]) for t in range(x.shape[self.T])], 2)
        x = self.activ(x)
        x = self.conv(x)
        
        x = torch.stack([self.drop(x[:,:,t,...]) for t in range(x.shape[self.T])], 2)
        
        return x



class CNNLayerTranspose_long(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 T:int=2,
                 kernel_size:[int, list]=3,
                 kernel_shape:str='hypercube',
                 norm_type:str=None,
                 activ_type:str=None,
                 drop_rate=0,
                 **kwargs
    ):
        super(CNNLayerTranspose_long, self).__init__()
        self.T = T
        
        drop = eval('nn.Dropout%dd' % X)
        norm = eval('nn.%sNorm%dd' % (norm_type, X))(n_input_features)
        activ = eval('nn.%s' % activ_type)() if activ_type is not None else nn.Identity()
        conv = Conv4d(in_channels=n_input_features,
                      out_channels=n_output_features,
                      kernel_size=kernel_size,
                      kernel_shape=kernel_shape,
                      padding=1,
                      X=X,
        )

        self.add_module('drop', drop if drop_rate > 0 else nn.Identity())
        self.add_module('norm', norm if norm_type is not None else nn.Identity())
        self.add_module('activ', activ if activ_type is not None else nn.Identity())
        self.add_module('conv', conv)


    def forward(self, x):
        x = torch.stack([self.drop(x[:,:,t,...]) for t in range(x.shape[self.T])], 2)
        x = torch.stack([self.norm(x[:,:,t,...]) for t in range(x.shape[self.T])], 2)
        x = self.activ(x)
        x = self.conv(x)

        return x




class _UNetBlock(nn.ModuleDict):
    def __init__(self,
                 X:int,
                 config:list[int],
                 norm_type:str,
                 activ_type:str,
                 layer_type:str,
                 T:int=2,
                 drop_rate=0,
                 skip=False,
                 **kwargs
    ):
        super(_UNetBlock, self).__init__()        
        n_layers = len(config) - 1
        
        for n in range(n_layers):
            growth = 1 + (skip and n == n_layers - 1)
            layer = eval('CNNLayer_%s' % layer_type)(n_input_features=config[n],
                                                     n_output_features=growth*config[n+1],
                                                     norm_type=norm_type,
                                                     activ_type=activ_type,
                                                     drop_rate=drop_rate,
                                                     X=X, T=T
            )
            self.add_module('ConvLayer%d' % (n + 1), layer)

            
    def forward(self, x):
        x_out = [None] * (len(self.items()) + 1)
        x_out[0] = x

        for n, [name, layer] in enumerate(self.items()):
            x_out[n+1] = layer(x_out[n])

        return x_out


    
class _UNetBlockTranspose(nn.ModuleDict):
    def __init__(self,
                 X:int,
                 config:list[int],
                 norm_type:str,
                 activ_type:str,
                 layer_type:str,
                 T:int=2,
                 drop_rate=0,
                 skip=True,
                 **kwargs
    ):
        super(_UNetBlockTranspose, self).__init__()
        n_layers = len(config) - 1
        
        for n in reversed(range(1, n_layers)):
            growth = 1 + (skip and n == n_layers - 1)
            layer = eval('CNNLayerTranspose_%s' % layer_type)(n_input_features=growth*config[n],
                                                              n_output_features=config[n],
                                                              norm_type=norm_type,
                                                              activ_type=activ_type,
                                                              drop_rate=drop_rate,
                                                              X=X, T=T,
            )
            self.add_module('ConvLayer%d' % (n + 1), layer)
            

    def forward(self, x):
        x_out = [None] * (len(self.items()) + 1)
        x_out[0] = x

        for n, [name, layer] in enumerate(self.items()):
            x_out[n+1] = layer(x_out[n])

        return x_out
   


### Classes to specify image dimensions ###
class UNet3D_long(UNetXD_Longitudinal):
    def __init__(self, in_channels, out_channels_unet, out_channels_cnet, **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels_unet=out_channels_unet,
                         out_channels_cnet=out_channels_cnet,
                         n_levels=3,
                         n_starting_features=24,
                         X=3, T=2,
                         **kwargs
	)


class UNet2D_long(UNetXD_Longitudinal):
    def __init__(self, in_channels, out_channels_unet, out_channels_cnet, **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels_unet=out_channels_unet,
                         out_channels_cnet=out_channels_cnet,
                         X=2, T=2,
                         **kwargs
	)
