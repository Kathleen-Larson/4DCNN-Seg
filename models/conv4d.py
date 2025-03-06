import torch
import torch.nn as nn
from torch.nn.modules.utils import _quadruple
import math
import torch.nn.functional as F

#mostly based off of https://github.com/ZhengyuLiang24/Conv4d-PyTorch/blob/main/Conv4d.py
class Conv4d(nn.Module):
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
                 kernel_shape='hypercube', #hypercube, hypercross, etc.
                 X:int=3,
                 **kwargs
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get/assert constructer args
        if isinstance(kernel_size, int): kernel_size = [kernel_size] * (X + 1)
        if isinstance(stride, int): stride = [stride] * (X + 1)
        if isinstance(padding, int): padding = [padding] * (X + 1)
        if isinstance(dilation, int): dilation = [dilation] * (X + 1)
        
        assert len(kernel_size) == (X + 1)
        assert len(stride) == (X + 1)
        assert len(padding) == (X + 1)
        assert len(dilation) == (X + 1)
        
        valid_padding_modes = {'zeros'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        
        # Store constructor arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.X = X
        self.t_dim = 2
        
        
        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

        
        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv_layers = torch.nn.ModuleList()

        for i in range(self.kernel_size[0]):
            # Define kernel
            if kernel_shape == 'hypercube':
                kernel_size = self.kernel_size[1::]
            elif kernel_shape == 'hypercross':
                kernel_size = self.kernel_size[1::] if i in [(self.kernel_size[0] - 1)//2] else 1

            # Initialize a Conv3D layer
            layer = eval('nn.Conv%dd' % X)(in_channels=self.in_channels, 
                                           out_channels=self.out_channels,
                                           kernel_size=kernel_size,
                                           padding=self.padding[1::],
                                           dilation=self.dilation[1::],
                                           stride=self.stride[1::],
                                           bias=False)
            layer.weight = nn.Parameter(self.weight[:, :, i, :, :])

            # Store the layer
            self.conv_layers.append(layer)

        del self.weight


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, x):
        if self.X == 3:
            (Batch, _, t_i, d_i, h_i, w_i) = tuple(x.shape)
            (t_k, h_k, w_k, d_k) = self.kernel_size
            (t_p, h_p, w_p, d_p) = self.padding
            (t_d, h_d, w_d, d_d) = self.dilation
            (t_s, h_s, w_s, d_s) = self.stride

            t_o = (t_i + 2 * t_p - (t_k) - (t_k-1) * (t_d-1))//t_s + 1
            out = [None] * t_o
            
            for i in range(t_k):
                zero_offset = - t_p + (i * t_d)
                j_start = max(zero_offset % t_s, zero_offset)
                j_end = min(t_i, t_i + t_p - (t_k-i-1)*t_d)

                for j in range(j_start, j_end, t_s):
                    out_frame = (j - zero_offset) // t_s
                    out[out_frame] = self.conv_layers[i](x[:,:,j,...]) if out[out_frame] is None \
                        else out[out_frame] + self.conv_layers[i](x[:,:,j,...])
            
            out = torch.stack(out, dim=self.t_dim)
            
            if self.bias is not None:
                out = out + self.bias.view(1,-1,1,1,1,1)

                
        elif self.X == 2:
            (Batch, _, t_i, h_i, w_i) = tuple(x.shape)
            (t_k, h_k, w_k) = self.kernel_size
            (t_p, h_p, w_p) = self.padding
            (t_d, h_d, w_d) = self.dilation
            (t_s, h_s, w_s) = self.stride

            t_o = (t_i + 2 * t_p - (t_k) - (t_k-1) * (t_d-1))//t_s + 1
            out = [None] * t_o

            for i in range(t_k):
                zero_offset = - t_p + (i * t_d)
                j_start = max(zero_offset % t_s, zero_offset)
                j_end = min(t_i, t_i + t_p - (t_k-i-1)*t_d)

                for j in range(j_start, j_end, t_s):
                    out_frame = (j - zero_offset) // t_s
                    out[out_frame] = self.conv_layers[i](x[:,:,j,...]) if out[out_frame] is None \
                        else out[out_frame] + self.conv_layers[i](x[:,:,j,...])

            out = torch.stack(out, dim=self.t_dim)
            
            if self.bias is not None:
                out = out + self.bias.view(1,-1,1,1,1)

        return out




class ConvTranspose4d(nn.Module):
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
                 kernel_shape='hypercube', #hypercube, hypercross, etc.
                 X:int=3,
                 **kwargs
    ):
        super().__init__()


        # Get/assert constructer args
        if isinstance(kernel_size, int): kernel_size = [kernel_size] * (X + 1)
        if isinstance(stride, int): stride = [stride] * (X + 1)
        if isinstance(padding, int): padding = [padding] * (X + 1)
        if isinstance(dilation, int): dilation = [dilation] * (X + 1)

        assert len(kernel_size) == (X + 1)
        assert len(stride) == (X + 1)
        assert len(padding) == (X + 1)
        assert len(dilation) == (X + 1)

        valid_padding_modes = {'zeros'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))


        # Store constructor arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.X = X
        self.T = 2


        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()


        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv_layers = torch.nn.ModuleList()

        for i in range(self.kernel_size[0]):
            # Define kernel
            if kernel_shape == 'hypercube':
                kernel_size = self.kernel_size[1::]
            elif kernel_shape == 'hypercross':
                kernel_size = self.kernel_size[1::] if i in [(self.kernel_size[0] - 1)//2] else 1

            # Initialize a Conv3D layer
            layer = eval('nn.ConvTranspose%dd' % X)(in_channels=self.in_channels,
                                                    out_channels=self.out_channels,
                                                    kernel_size=kernel_size,
                                                    padding=self.padding[1::],
                                                    dilation=self.dilation[1::],
                                                    stride=self.stride[1::],
                                                    bias=False)
            layer.weight = nn.Parameter(self.weight[:, :, i, :, :])

            # Store the layer
            self.conv_layers.append(layer)

        del self.weight



    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)



    def forward(self, x, out_shape=None):
        if self.X == 3:
            (Batch, _, t_i, d_i, h_i, w_i) = tuple(x.shape)
            (t_k, h_k, w_k, d_k) = self.kernel_size
            (t_p, h_p, w_p, d_p) = self.padding
            (t_d, h_d, w_d, d_d) = self.dilation
            (t_s, h_s, w_s, d_s) = self.stride

            t_o = (t_i - 1) * t_s - 2 * t_p - t_d * (t_k - 1) + 1
            out = [None] * t_o

            for i in range(t_k):
                zero_offset = - t_p + (i * t_d)
                j_start = max(zero_offset % t_s, zero_offset)
                j_end = min(t_i, t_i + t_p - (t_k-i-1)*t_d)

                for j in range(j_start, j_end, t_s):
                    out_frame = (j - zero_offset) // t_s
                    out[out_frame] = self.conv_layers[i](x[:,:,j,...]) if out[out_frame] is None \
                        else out[out_frame] + self.conv_layers[i](x[:,:,j,...])

            out = torch.stack(out, dim=self.T)

            if self.bias is not None:
                out = out + self.bias.view(1,-1,1,1,1,1)


        elif self.X == 2:
            (Batch, _, t_i, h_i, w_i) = tuple(x.shape)
            (t_k, h_k, w_k) = self.kernel_size
            (t_p, h_p, w_p) = self.padding
            (t_d, h_d, w_d) = self.dilation
            (t_s, h_s, w_s) = self.stride

            t_o = (t_i - 1) * t_s - 2 * t_p - t_d * (t_k - 1) + 1
            out = [None] * t_o
            
            for i in range(t_k):
                zero_offset = - t_p + (i * t_d)
                j_start = max(zero_offset % t_s, zero_offset)
                j_end = min(t_i, t_i + t_p - (t_k-i-1)*t_d)

                for j in range(j_start, j_end, t_s):
                    out_frame = (j - zero_offset) // t_s
                    out[out_frame] = self.conv_layers[i](x[:,:,j,...]) if out[out_frame] is None \
                        else out[out_frame] + self.conv_layers[i](x[:,:,j,...])
                    
            out = torch.stack(out, dim=self.T)

            if self.bias is not None:
                out = out + self.bias.view(1,-1,1,1,1)

        return out
    
