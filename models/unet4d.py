import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from conv4d import Conv4d


### 4D UNet Model

class UNet4D(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 block_config:tuple,
                 n_convs_per_block:int,
                 kernel_size:[int, tuple].
                 kernel_shape:str):
        super().__init__()

        # Initialize
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = len(block_config)
        self.n_convs_per_block = n_convs_per_block
        self.activation_type = activation_type
        self.pooling_type = pooling_type
        
        self.blocks = nn.Sequential()
        
        
        # Define encoding blocks (conv layer + pooling)
        n_input_features = in_channels
        
        for b in range(0, n_blocks):
            n_output_features = block_config[b]
            if b == n_blocks - 1:
                block = _UNet4D_DownBlock(n_input_features, n_output_features,
                                          n_convs_per_block, kernel_size, kernel_shape,
                                          pool=False)
            else:
                block = _UNet4D_DownBlock(n_input_features, n_output_features,
                                          n_convs_per_block, kernel_size, kernel_shape)
            self.blocks.add_module('down%d' % (b+1), block)
            n_input_features = n_output_features
            

        # Define decoding blocks (upsample + conv layer)
        n_input_features = block_config[-1]
        
        for b in range(0, n_blocks):
            n_output_features = block_config[n_blocks - (b+1)]
            if b == n_blocks - 1:
                block = _UNet4D_UpBlock(n_output_features, n_output_features
                                   kernel_size, kernel_shape)
            else:
                
                block = _UNet4D_UpBlock(n_input_features, n_output_features
                                   kernel_size, kernel_shape)
            self.blocks_upsample.add_module('up%d' % (b+1), block)
            n_input_features = n_output_features
            

    def forward(self, img):
        encoding = [None] * self.n_blocks
        decoding = [None] * self.n_blocks
        skips = [None] * (self.n_blocks - 1)
        
        # Down
        encoding[0] = x
        for b in range(0, self.n_blocks):
            #print("Down", b+1)
            #print("input:", encoding[b].shape)
            if b != self.n_blocks - 1:
                encoding[b+1], skips[b] = self.blocks.__getattr__('down%d' % (b+1))(encoding[b])
            else:
                encoding[b+1], _ = self.blocks.__getattr__('down%d' % (b+1))(encoding[b])
            #print("output:", encoding[b+1].shape)
            #print(" ")

        # Up
        decoding[0] = encoding[-1]
        for b in range(0, self.n_blocks):
            print("Up", b+1)
            print("input:", decoding[b].shape)
            if b != self.n_blocks - 1:
                skip_ind = self.n_blocks - b - 2
                print("skip:", skips[skip_ind].shape)
                decoding[b+1] = self.blocks.__getattr__('up%d' % (b+1))(decoding[b], skips[skip_ind])
            else:
                decoding[b+1] = self.blocks.__getattr__('up%d' % (b+1))(decoding[b])
            print("output:", decoding[b+1].shape)
            print(" ")

        return decoding[-1]




# Downsampling (encoding) block
class _UNet4D_DownBlock(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 n_convs_per_block:int,
                 kernel_shape:int,
                 activation_type:str,
                 pooling_type:str,
                 pool=True):
        super(_UNet4D_DownBlock, self).__init__()
        self.n_convs = n_convs_per_block
        self.pool = pool
        
        # Define components
        #self.pooling = ??
        for n in range(0, n_convs_per_block):
            if n == 0:
                layer = nn.Sequential(Conv4d(in_channels, out_channels, kernel_size, kernel_shape),
                                      getattr(nn, activation_type)(inplace=True))
            else:
                layer = nn.Sequential(Conv4d(out_channels, out_channels, kernel_size, kernel_shape),
                                      getattr(nn, activation_type)(inplace=True))
            self.conv_block.add_module('convlayer%d' % (n+1), layer))


    def forward(self, x):
        for n in range(0, self.n_convs):
            x = self.conv_block.__getattr__('convlayer%d' % (n+1))(x)

        if self.pool:
            return self.pooling(x), x
        else:
            return x, x
                            


# Define the upsampling (decoding) block
class _UNet4D_UpBlock(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 n_convs_per_block:int,
                 activation_type:str):
        super(_UNet4D_UpBlock, self).__init__()
        self.n_convs = n_convs_per_block
        
        # Define components
        self.up = Conv4d(in_channels, out_channels, kernel_size=2, kernel_shape=2)
        self.conv_block = nn.Sequential()

        for n in range(0,n_convs_per_block):
            if n == 0:
                layer = nn.Sequential(Conv4d(in_channels, out_channels, kernel_size=3, padding=1),
                                      getattr(nn, activation_type)(inplace=True))
            else:
                layer = nn.Sequential(Conv4d(out_channels, out_channels, kernel_size=3, padding=1),
                                      getattr(nn, activation_type)(inplace=True))
            self.conv_block.add_module('convlayer%d' % (n+1), layer)


    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)

        for n in range(0, self.n_convs):
            x = self.conv_block.__getattr__('convlayer%d' % (n+1))(x)

        return x


