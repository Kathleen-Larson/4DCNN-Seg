import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from .conv4d import Conv4d
from .conv4d_transpose import ConvTranspose4d


class UNetXD_Longitudinal(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 n_convs_per_block:int=2,
                 feature_size_config:list[int]=(24, 48, 96, 192),
                 temporal_config:list[int]=(0, 0, 0, 0),
                 kernel_size:[int, tuple]=3,
                 kernel_shape:str='hypercube',
                 normalization_type:str='Instance',
                 activation_type:str='ReLU',
                 skip:bool=True,
                 X:int=3,
                 T:int=2,
                 **kwargs
    ):
        super(UNetXD_Longitudinal, self).__init__()

        self.X = X # number of image dims
        self.T = T # temporal dimension
        
        self.feature_size_config = list(feature_size_config)
        self.temporal_config = list(temporal_config)

        assert len(self.feature_size_config) == len(self.temporal_config), \
            "len(self.feature_size_config should equal len(self.temporal_config)"

        self.n_blocks = len(feature_size_config)
        self.skip = skip

        self.activation_type = activation_type if callable(getattr(nn, activation_type)) \
            else Exception("Invalid activation_type (not an attribute of torch.nn")

        
        # Encoding blocks:
        self.encoding = nn.Sequential()
        encoding_config = [in_channels] + self.feature_size_config
        
        for b in range(len(encoding_config)-1):
            block = _UNetBlock(n_input_features=encoding_config[b],
                               n_output_features=encoding_config[b+1],
                               layer_type='long' if temporal_config[b] else 'standard',
                               n_layers=n_convs_per_block,
                               norm_type=normalization_type,
                               activ_type=activation_type,
                               level=b,
                               X=self.X,
                               T=self.T
            )
            self.encoding.add_module('ConvBlock%d' % (b+1), block)

            downsample = _DownSample(n_input_features=encoding_config[b+1],
                                     n_output_features=encoding_config[b+1],
                                     factor=2,
                                     X=self.X,
                                     temporal=True if self.temporal_config[b] else False
            )
            self.encoding.add_module('Downsample%d' % (b+1), downsample)
                
            """
            pool = _MaxPool(X=self.X,
                            T=self.T,
                            temporal=False, #True if temporal_config[b] else False,
                            spatial_kernel=2,
                            spatial_stride=2,
                            temporal_kernel=2,
                            temporal_stride=2
            )
            self.encoding.add_module('Pool%d' % (b+1), pool)
            """
            
        #Decoding blocks:
        self.decoding = nn.Sequential()
        decoding_config = [out_channels] + self.feature_size_config
        
        for b in reversed(range(len(decoding_config) - 1)):
            upsample = _UpSample(n_input_features=decoding_config[b+1],
                                     n_output_features=decoding_config[b+1],
                                     factor=2,
                                     X=self.X,
                                     temporal=True if self.temporal_config[b] else False
            )
            self.decoding.add_module('Upsample%d' % (b+1), upsample)
            """
            unpool = _MaxUnpool(X=self.X,
                                T=self.T,
                                temporal=False, #True if temporal_config[b] else False,
                                spatial_kernel=2,
                                spatial_stride=2,
                                temporal_kernel=2,
                                temporal_stride=2
            )
            self.decoding.add_module('Unpool%d' % (b+1), unpool)
            """
            block = _UNetBlockTranspose(n_input_features=decoding_config[b+1],
                                        n_output_features=decoding_config[b],
                                        layer_type='long' if temporal_config[b] else 'standard',
                                        n_layers=n_convs_per_block,
                                        norm_type=normalization_type,
                                        activ_type=activation_type,
                                        level=b,
                                        drop_rate=0,
                                        X=self.X,
                                        T=self.T
            )
            self.decoding.add_module('ConvBlock%d' % (b+1), block)
    

    def forward(self, x):
        enc = [None] * (self.n_blocks)     # encoding
        dec = [None] * (self.n_blocks)     # decoding
        siz = [None] * (self.n_blocks)     # maxunpool output size
        
        # Encoding
        for b in range(self.n_blocks):
            x = enc[b] = self.encoding.__getattr__('ConvBlock%d' % (b+1))(x)
            siz[b] = x.shape
            if b != self.n_blocks - 1:
                x = self.encoding.__getattr__('Downsample%d' % (b+1))(x)
                #x, i_s[b], i_t[b] = self.encoding.__getattr__('Pool%d' % (b+1))(x)

        # Decoding
        for b in reversed(range(self.n_blocks)):
            if b != self.n_blocks - 1:
                x = self.decoding.__getattr__('Upsample%d' % (b+1))(x, siz[b])
                #x = self.decoding.__getattr__('Unpool%d' % (b+1))(x, i_s[b], i_t[b], siz[b])
            x = dec[b] = self.decoding.__getattr__('ConvBlock%d' % (b+1))(torch.cat([x, enc[b]], 1))

        return x



class _DownSample(nn.Module):
    def __init__(self,
                 n_input_features:int,
                 n_output_features:int,
                 factor:list[int],
                 X:int,
                 T:int=2,
                 temporal:bool=True,
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
                 n_input_features:int,
                 n_output_features:int,
                 factor:list[int],
                 X:int,
                 T:int=2,
                 temporal:bool=True,
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
        """
        upconv = ConvTranspose4d(in_channels=n_input_features,
                                 out_channels=n_output_features,
                                 kernel_size=factor,
                                 stride=factor,
                                 X=self.X,
                                 T=self.T,
        ) if self.temporal else eval('nn.ConvTranspose%dd' % X)(in_channels=n_input_features,
                                                                out_channels=n_output_features,
                                                                kernel_size=factor,
                                                                stride=factor,
        )
        """
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
    

    
class _UNetLayer_standard(nn.Module):
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
        super(_UNetLayer_standard, self).__init__()
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


    
class _UNetLayerTranspose_standard(nn.Module):
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
        super(_UNetLayerTranspose_standard, self).__init__()
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




class _UNetLayer_long(nn.Module):
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
        super(_UNetLayer_long, self).__init__()
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



class _UNetLayerTranspose_long(nn.Module):
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
        super(_UNetLayerTranspose_long, self).__init__()
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
                 n_input_features:int,
                 n_output_features:int,
                 n_layers:int,
                 norm_type:str,
                 activ_type:str,
                 level:int,
                 layer_type:str,
                 T:int=2,
                 drop_rate=0,
                 skip=False,
                 **kwargs
    ):
        super(_UNetBlock, self).__init__()        
        layer = eval('_UNetLayer_%s' % layer_type)(n_input_features=n_input_features,
                                                   n_output_features=n_output_features,
                                                   norm_type=norm_type,
                                                   activ_type=activ_type if level != 0 else None,
                                                   drop_rate=drop_rate,
                                                   X=X, T=T
        )
        self.add_module('ConvLayer1', layer)

        for i in range(1, n_layers):
            growth = 1 + (skip and i == n_layers - 1)
            layer = eval('_UNetLayer_%s' % layer_type)(n_input_features=n_output_features,
                                                       n_output_features=growth*n_output_features,
                                                       norm_type=norm_type,
                                                       activ_type=activ_type,
                                                       drop_rate=drop_rate,
                                                       X=X, T=T
            )
            self.add_module('ConvLayer%d' % (i + 1), layer)

            
    def forward(self, x):
        for name, layer in self.items():
            x = layer(x)
        return x


    
class _UNetBlockTranspose(nn.ModuleDict):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 n_layers:int,
                 norm_type:str,
                 activ_type:str,
                 level:int,
                 layer_type:str,
                 T:int=2,
                 drop_rate=0,
                 skip=True,
                 **kwargs
    ):
        super(_UNetBlockTranspose, self).__init__()
        for i in reversed(range(1,n_layers)):
            growth = 1 + (skip and i == n_layers - 1)
            layer = eval('_UNetLayerTranspose_%s' % layer_type)(n_input_features=growth*n_input_features,
                                                                n_output_features=n_input_features,
                                                                norm_type=norm_type,
                                                                activ_type=activ_type,
                                                                drop_rate=drop_rate,
                                                                X=X, T=T,
            )
            self.add_module('ConvLayer%d' % (i + 1), layer)
            
        layer = eval('_UNetLayerTranspose_%s' % layer_type)(n_input_features=n_input_features,
                                                            n_output_features=n_output_features,
                                                            norm_type=norm_type,
                                                            activ_type=activ_type,
                                                            drop_rate=drop_rate,
                                                            X=X, T=T,
        )
        self.add_module('ConvLayer1', layer)


    def forward(self, x):
        for name, layer in self.items():
            x = layer(x)
        return x
   


### Classes to specify image dimensions ###
class UNet3D_long(UNetXD_Longitudinal):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         n_convs_per_block=2,
                         feature_size_config=(24, 48, 96),
                         temporal_config=(1, 1, 0),
                         kernel_size=(3, 3, 3, 3),
                         kernel_shape='hypercube',
                         normalization_type='Instance',
                         activation_type='ReLU',
                         skip=True,
                         X=3,
                         T=2,
                         **kwargs
	)
        
        
        
class UNet2D_long(UNetXD_Longitudinal):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         n_convs_per_block=2,
                         feature_size_config=(24, 48, 96),
                         temporal_config=(1, 0, 0),
                         kernel_size=(3, 3, 3),
                         kernel_shape='hypercube',
                         normalization_type='Instance',
                         activation_type='ReLU',
                         skip=True,
                         X=2,
                         T=2,
                         **kwargs
	)
