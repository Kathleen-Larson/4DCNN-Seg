import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv4d import Conv4d


class UNetXD_Longitudinal(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 n_convs_per_block:int=2,
                 block_config:tuple=(24, 48, 96, 192),
                 kernel_size:[int, tuple]=3,
                 kernel_shape:str='hypercube',
                 normalization_type:str='Instance',
                 activation_type:str='ReLU',
                 skip:bool=True,
                 X:int=3,
                 **kwargs
    ):
        super(UNetXD_Longitudinal, self).__init__()

        self.X = X
        self.block_config = list(block_config)
        self.n_blocks = len(block_config)
        self.skip = skip

        self.activation_type = activation_type if callable(getattr(nn, activation_type)) \
            else Exception("Invalid activation_type (not an attribute of torch.nn")

        
        # Encoding blocks:
        self.encoding = nn.Sequential()
        encoding_config = [in_channels] + self.block_config
        
        for b in range(0, len(encoding_config)-1):
            block = _UNetBlock(n_input_features=encoding_config[b],
                               n_output_features=encoding_config[b+1],
                               layer_type='long',
                               n_layers=n_convs_per_block,
                               norm_type=normalization_type,
                               activ_type=activation_type,
                               level=b,
                               drop_rate=0,
                               X=X,
            )
            self.encoding.add_module('ConvBlock%d' % (b+1), block)
            pool = eval('nn.MaxPool%dd' % X)(kernel_size=2,
                                             stride=2,
                                             return_indices=True
            )
            self.encoding.add_module('Pool%d' % (b+1), pool)

            
        #Decoding blocks:
        self.decoding = nn.Sequential()
        decoding_config = [out_channels] + self.block_config
        
        for b in reversed(range(0, len(decoding_config) - 1)):
            upsample = eval('nn.MaxUnpool%dd' % X)(kernel_size=2, stride=2)
            self.decoding.add_module('Upsample%d' % (b+1), upsample)
            
            block = _UNetBlockTranspose(n_input_features=decoding_config[b+1],
                                        n_output_features=decoding_config[b],
                                        layer_type='long',
                                        n_layers=n_convs_per_block,
                                        norm_type=normalization_type,
                                        activ_type=activation_type,
                                        level=b,
                                        drop_rate=0,
                                        X=X,
            )
            self.decoding.add_module('ConvBlock%d' % (b+1), block)


    def _pooling_iterator(self, x, idx):
        x_pool = [None] * x.shape[2]
        inds = [None] * x.shape[2]

        for t in range(x.shape[2]):
            x_pool[t], inds[t] = self.encoding.__getattr__('Pool%d' % (idx))(x[:,:,t,...])

        x_pool = torch.stack([x_pool[t] for t in range(x.shape[2])], 2)
        inds = torch.stack([inds[t] for t in range(x.shape[2])], 2)
        return x_pool, inds


    def _unpooling_iterator(self, x, inds, siz, idx):
        x_unpool = [None] * x.shape[2]
        siz = list(siz[0:2]) + list(siz[3:])

        for t in range(x.shape[2]):
            x_unpool[t] = self.decoding.__getattr__('Upsample%d' % (idx))(x[:,:,t,...],
                                                                          inds[:,:,t,...],
                                                                          output_size=siz)
        x_unpool = torch.stack([x_unpool[t] for t in range(x.shape[2])], 2)
        return x_unpool
    

    def forward(self, x):
        enc = [None] * (self.n_blocks) # encoding
        dec = [None] * (self.n_blocks) # decoding
        idx = [None] * (self.n_blocks) # maxpool indices
        siz = [None] * (self.n_blocks) # maxunpool output size
        
        # Encoding
        for b in range(0, self.n_blocks):
            x = enc[b] = self.encoding.__getattr__('ConvBlock%d' % (b+1))(x)
            siz[b] = x.shape
            if b != self.n_blocks - 1:
                x, idx[b] = self._pooling_iterator(x, b+1)

            #breakpoint()
                
        # Decoding
        for b in reversed(range(0, self.n_blocks)):
            if b != self.n_blocks - 1:
                x = self._unpooling_iterator(x, idx[b], siz[b], b+1)
            x = dec[b] = self.decoding.__getattr__('ConvBlock%d' % (b+1))(torch.cat([x, enc[b]], 1))
        
        return x


            
        
class _UNetLayer_standard(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 norm_type:str=None,
                 activ_type:str=None,
                 drop_rate=0,
                 **kwargs
    ):
        super(_UNetLayer_standard, self).__init__()

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
        x = torch.stack([self.norm(x[...,t,:,:]) for t in range(2)], 2)
        x = self.activ(x)
        x = torch.stack([self.conv(x[...,t,:,:]) for t in range(2)], 2)
        x = torch.stack([self.drop(x[...,t,:,:]) for t in range(2)], 2)
        
        return x


    
class _UNetLayerTranspose_standard(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 norm_type:str=None,
                 activ_type:str=None,
                 drop_rate=0,
                 **kwargs
    ):
        super(_UNetLayerTranspose_standard, self).__init__()
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
        x = torch.stack([self.drop(x[...,t,:,:]) for t in range(2)], 2)
        x = torch.stack([self.norm(x[...,t,:,:]) for t in range(2)], 2)
        x = self.activ(x)
        x = torch.stack([self.conv(x[...,t,:,:]) for t in range(2)], 2)
        
        return x




class _UNetLayer_long(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 kernel_size:[int, list]=3,
                 kernel_shape:str='hypercube',
                 norm_type:str=None,
                 activ_type:str=None,
                 drop_rate=0,
                 **kwargs
    ):
        super(_UNetLayer_long, self).__init__()

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
        x = torch.stack([self.norm(x[...,t,:,:]) for t in range(2)], 2)
        x = self.activ(x)
        x = self.conv(x) #torch.stack([self.conv(x[...,t,:,:]) for t in range(2)], 2)
        x = torch.stack([self.drop(x[...,t,:,:]) for t in range(2)], 2)

        return x



class _UNetLayerTranspose_long(nn.Module):
    def __init__(self,
                 X:int,
                 n_input_features:int,
                 n_output_features:int,
                 kernel_size:[int, list]=3,
                 kernel_shape:str='hypercube',
                 norm_type:str=None,
                 activ_type:str=None,
                 drop_rate=0,
                 **kwargs
    ):
        super(_UNetLayerTranspose_long, self).__init__()

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
        x = torch.stack([self.drop(x[...,t,:,:]) for t in range(2)], 2)
        x = torch.stack([self.norm(x[...,t,:,:]) for t in range(2)], 2)
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
                                                   X=X,
        )
        self.add_module('ConvLayer1', layer)

        for i in range(1, n_layers):
            growth = 1 + (skip and i == n_layers - 1)
            layer = eval('_UNetLayer_%s' % layer_type)(n_input_features=n_output_features,
                                                       n_output_features=growth*n_output_features,
                                                       norm_type=norm_type,
                                                       activ_type=activ_type,
                                                       drop_rate=drop_rate,
                                                       X=X,
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
                                                                X=X,
            )
            self.add_module('ConvLayer%d' % (i + 1), layer)
            
        layer = eval('_UNetLayerTranspose_%s' % layer_type)(n_input_features=n_input_features,
                                                            n_output_features=n_output_features,
                                                            norm_type=norm_type,
                                                            activ_type=activ_type,
                                                            drop_rate=drop_rate,
                                                            X=X,
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
                         block_config=(24, 48, 96, 192),
                         kernel_size=(3, 3, 3, 3),
                         kernel_shape='hypercross',
                         normalization_type='Instance',
                         activation_type='ELU',
                         skip=True,
                         X=3,
                         **kwargs
	)
        
        
        
class UNet2D_long(UNetXD_Longitudinal):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         n_convs_per_block=2,
                         block_config=(24, 48, 96, 192),
                         kernel_size=(3, 3, 3),
                         kernel_shape='hypercube',
                         normalization_type='Instance',
                         activation_type='ELU',
                         skip=True,
                         X=2,
                         **kwargs
	)
