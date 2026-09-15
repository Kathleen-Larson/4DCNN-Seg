import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
import utils


# --------------------------------------------------------------------------------------------------

class UNetLong(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            activ_func='ELU',
            conv_kernel_size=3,
            conv_kernel_shape=None,
            down_func='maxpool',
            down_kernel_size=2,
            dropout_rate=0,
            feature_ratio=1.2,
            n_convs_per_block=2,
            n_levels=4,
            n_starting_features=64,
            norm_func='InstanceNorm',
            return_multiple=True,
            return_second_to_last_output=False,
            skip_last_level=False,
            use_residuals=False,
            use_skips=True,
            L=0,
            T=2,
            X=3,
            **kwargs
    ):
        super().__init__()

        # Parse args
        self.L = L
        self.X = X
        self.T = T

        self.in_channels = in_channels * self.T
        self.out_channels = out_channels

        self.n_starting_features = n_starting_features
        self.n_levels = n_levels
        self.return_multiple = return_multiple
        self.return_second_to_last_output = return_second_to_last_output
        self.skip_last_level = skip_last_level
        self.use_residuals = use_residuals
        self.use_skips = use_skips

        conv_block_kwargs = {
            'activ_func': _parse_arg_as_function(self, activ_func),
            'norm_func': _parse_arg_as_function(self, norm_func),
            'conv_kernel_size': _parse_arg_as_list(self, conv_kernel_size, int),
            'dropout_rate': dropout_rate,
            'X': self.X,
            'T': self.T
        }

        f_config = [
            int(n_starting_features * (feature_ratio ** n)) for n in range(self.n_levels)
        ]
        enc = [self.in_channels] + f_config[:-1]
        dec = f_config
        self.n_transfer_features = f_config[self.L]

        # Encoding block
        self.enc_block = nn.ModuleList(
            [_ConvBlock(
                f_config=([enc[n]] + ([enc[n + 1]] * n_convs_per_block)),
                **conv_block_kwargs
            ) for n in range(self.n_levels - 1)]
        )
        self.downsample = nn.ModuleList(
            [_Pool(
                n_features=enc[n + 1],
                pool_type='Max',
                kernel_size=down_kernel_size,
                X=self.X,
                T=self.T
            ) for n in range(self.n_levels - 1)]
        )
        self.bottleneck = _ConvBlock(
            f_config=([enc[-1]] + [dec[-1]] * n_convs_per_block),
            **conv_block_kwargs
        )

        # Decoding arm        
        self.dec_block = nn.ModuleList(
            [_ConvBlock(
                f_config=(
                    [dec[n + 1] + enc[n + 1] if self.use_skips else dec[n + 1]]
                    + ([dec[n]] * n_convs_per_block)
                ),
                **conv_block_kwargs
            ) for n in range(self.n_levels - 1)]
        )
        self.upsample = nn.ModuleList(
            [_UpConv(
                n_features=dec[n + 1],
                kernel_size=down_kernel_size,
                X=X, T=T
            ) for n in range(self.n_levels - 1)]
        )

        # Final layer
        self.final = _ConvBlock(
            f_config=[n_starting_features, self.out_channels * self.T],
            conv_kernel_size=([1] * (self.X + 1)),
            T=self.T, X=self.X
        )

    def forward(self, x):
        B = x.shape[0]
        HWD = x.shape[-self.X:]

        x = x.view(B, -1, *HWD)
        y = None

        skips = [None] * (self.n_levels - 1)
        siz = [None] * (self.n_levels - 1)

        # Encoding arm
        for n in range(self.n_levels - 1):
            x = self.enc_block[n](x)
            skips[n], siz[n] = (x, x.shape)
            x = self.downsample[n](x)

        x = self.bottleneck(x)

        # Decoding arm
        for n in reversed(range(1 if self.skip_last_level else 0, self.n_levels - 1)):
            x = self.upsample[n](x, siz[n])
            x = self.dec_block[n](torch.cat([skips[n], x], dim=1) if self.use_skips else x)
            if n == self.L and self.return_multiple:
                y = self.upsample[n - 1](x, siz[n - 1])

        # Final layer
        if self.skip_last_level:
            return self.upsample[0](x, siz[0]), skips[0]

        y = x if y is None and self.return_multiple else y
        x = self.final(x).view(B, self.out_channels, self.T, *HWD)
        return x if y is None else (x, y)


class CNetLong(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            out_classes,
            in_shape,
            activ_func='ELU',
            conv_kernel_size=3,
            conv_kernel_shape=None,
            do_MLP=False,
            down_func='maxpool',
            down_kernel_size=2,
            dropout_rate=0,
            feature_ratio=1.2,
            model_type=3,
            n_convs_per_block=2,
            n_levels=4,
            n_starting_features=64,
            norm_func='InstanceNorm',
            L=0,
            X=3,
            T=2,
            **kwargs
    ):
        super().__init__()

        self.L = L
        self.X = X
        self.T = T

        self.in_channels = in_channels
        self.out_classes = out_classes
        self.out_channels = out_channels

        self.do_MLP = do_MLP
        self.model_type = model_type
        self.n_levels = n_levels
        self.use_residuals = False
        self.use_skips = True

        conv_block_kwargs = {
            'activ_func': _parse_arg_as_function(self, activ_func),
            'norm_func': _parse_arg_as_function(self, norm_func),
            'conv_kernel_size': _parse_arg_as_list(self, conv_kernel_size, int),
            'dropout_rate': dropout_rate,
            'X': self.X,
            'T': self.T
        }
        f_config = [
            math.ceil(in_channels * (feature_ratio ** n)) for n in range(self.n_levels + 1)
        ]
        enc = f_config[:-1]
        dec = f_config[1:]

        # Encoding block
        self.enc1_block = nn.ModuleList(
            [_ConvBlock(
                f_config=([enc[n]] + ([enc[n + 1]] * n_convs_per_block)),
                **conv_block_kwargs
            ) for n in range(self.n_levels - 1)]
        )
        self.downsample1 = nn.ModuleList(
            [_Pool(
                n_features=enc[n + 1],
                pool_type='Max',
                kernel_size=down_kernel_size,
                X=self.X,
                T=self.T
            ) for n in range(self.n_levels - 1)]
        )
        self.bottleneck1 = _ConvBlock(
            f_config=([enc[-1]] + [dec[-1]] * n_convs_per_block),
            **conv_block_kwargs
        )

        # Decoding arm
        if self.model_type > 1:
            self.dec_block = nn.ModuleList(
                [_ConvBlock(
                    f_config=(
                        [dec[n + 1] + enc[n + 1] if self.use_skips else dec[n + 1]]
                        + ([dec[n]] * n_convs_per_block)
                    ),
                    **conv_block_kwargs
                ) for n in range(self.n_levels - 1)]
            )
            self.upsample = nn.ModuleList(
                [_UpConv(
                    n_features=dec[n + 1],
                    kernel_size=down_kernel_size,
                    X=X, T=T
                ) for n in range(self.n_levels - 1)]
            )
            self.bottleneck2 = _ConvBlock(
                f_config=([dec[0]] + [enc[0]] * n_convs_per_block),
                **conv_block_kwargs
            )

        # Encoding arm 2
        if self.model_type > 2:
            self.enc2_block = nn.ModuleList(
                [_ConvBlock(
                    f_config=([enc[n]] + ([enc[n + 1]] * n_convs_per_block)),
                    **conv_block_kwargs
                ) for n in range(self.n_levels - 1)]
            )
            self.downsample2 = nn.ModuleList(
                [_Pool(
                    n_features=enc[n + 1],
                    pool_type='Max',
                    kernel_size=down_kernel_size,
                    X=self.X,
                    T=self.T
                ) for n in range(self.n_levels - 1)]
            )
            self.bottleneck3 = _ConvBlock(
                f_config=([enc[-1]] + [dec[-1]] * n_convs_per_block),
                **conv_block_kwargs
            )

        # Final layers
        self.change_map = _ConvBlock(
            f_config=[enc[0], out_channels],
            conv_kernel_size=([1] * (X + 1)),
            T=T, X=X
        )

        self.flatten = torch.nn.Flatten(1, -1)

        in_shape = in_shape if isinstance(in_shape, (list, tuple)) else in_shape ** self.X
        final_shape = [x // (2 ** len(self.downsample1)) for x in in_shape]
        n_in = math.prod(final_shape) * f_config[-1]

        if self.do_MLP:
            self.final_pool = _Pool(
                n_features=enc[-1],
                pool_type='AdaptiveAvg',
                out_shape=(1, 1, 1),
                kernel_size=[int(x / (2 ** (4))) for x in in_shape],
                X=self.X,
                T=self.T
            )
            self.MLP = _MLPBlock(
                f_config=[dec[-1], 128, n_starting_features, out_classes],
                activ_func=nn.GELU,  # conv_block_kwargs.get('activ_func'),
                norm_func=eval(f'nn.{norm_func}1d'),
                do_softmax=False,
            )
        else:
            self.final = _LinearBlock(
                f_config=([n_in, out_classes]),
                activ_func=conv_block_kwargs.get('activ_func'),
                conv_size=conv_block_kwargs.get('conv_kernel_size')[0],
            )

    def forward(self, x):
        B = x.shape[0]
        HWD = x.shape[-self.X:]

        skips = [None] * (self.n_levels - 1)
        siz = [None] * (self.n_levels - 1)

        # Single encoder
        if self.model_type == 1:
            y = self.change_map(x).view(B, -1, (self.T - 1), *HWD)

            for n in range(self.n_levels - 1):
                x = self.enc1_block[n](x)
                skips[n], siz[n] = (x, x.shape)
                x = self.downsample1[n](x)

            x = self.bottleneck1(x)

        # Encoder + decoder
        if self.model_type == 2:
            for n in range(self.n_levels - 1):
                x = self.enc1_block[n](x)
                skips[n], siz[n] = (x, x.shape)
                x = self.downsample1[n](x)

            x = self.bottleneck1(x)
            y = x

            for n in reversed(range(self.n_levels - 1)):
                y = self.upsample[n](y, siz[n])
                y = self.dec_block[n](torch.cat([skips[n], y], dim=1) if self.use_skips else y)

            y = self.bottleneck2(y)
            y = self.change_map(y).view(B, -1, (self.T - 1), *HWD)

        # Encoder + decoder + second encoder
        if self.model_type == 3:
            for n in range(self.n_levels - 1):
                x = self.enc1_block[n](x)
                skips[n], siz[n] = (x, x.shape)
                x = self.downsample1[n](x)

            x = self.bottleneck1(x)

            for n in reversed(range(self.n_levels - 1)):
                x = self.upsample[n](x, siz[n])
                x = self.dec_block[n](torch.cat([skips[n], x], dim=1) if self.use_skips else x)

            x = self.bottleneck2(x)
            y = self.change_map(x).view(B, -1, (self.T - 1), *HWD)

            for n in range(self.n_levels - 1):
                x = self.enc2_block[n](x)
                x = self.downsample2[n](x)

            x = self.bottleneck3(x)

        # Final classifier layers
        if self.do_MLP:
            x = self.final_pool(x)
            while x.shape[-1] == 1:
                x = x.squeeze(dim=-1)
            x = self.MLP(x)
        else:
            x = self.flatten(x)
            x = self.final(x)

        return x, y


class FineTuneLayers(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            activ_func='ELU',
            conv_kernel_size=3,
            conv_kernel_shape=None,
            dropout_rate=0,
            n_convs=2,
            n_skip_channels=0,
            norm_func='InstanceNorm',
            feature_ratio=1.2,
            return_multiple=True,
            use_residuals=False,
            T=2,
            X=3,
            **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.return_multiple = return_multiple
        self.X = X
        self.T = T

        conv_block_kwargs = {
            'activ_func': _parse_arg_as_function(self, activ_func),
            'norm_func': _parse_arg_as_function(self, norm_func),
            'conv_kernel_size': _parse_arg_as_list(self, conv_kernel_size, int),
            'dropout_rate': dropout_rate,
            'use_residuals': use_residuals,
            'X': self.X,
            'T': self.T
        }

        mid_n_channels = int(self.in_channels / feature_ratio) + 1

        self.conv_block = _ConvBlock(
            f_config=(
                [self.in_channels + n_skip_channels]
                + [int(self.in_channels / feature_ratio) + 1] * n_convs
            ), **conv_block_kwargs
        )

        self.final = _ConvBlock(
            f_config=[mid_n_channels, self.out_channels * self.T],
            conv_kernel_size=([1] * (self.X + 1)),
            T=self.T, X=self.X
        )

    def forward(self, x, skip_conn=None):
        B = x.shape[0]
        HWD = x.shape[-self.X:]

        # Run layers
        x = self.conv_block(torch.cat([skip_conn, x], dim=1) if skip_conn is not None else x)
        y = x if self.return_multiple else None

        # Final conv
        y = x if y is None and self.return_multiple else y
        x = self.final(x).view(B, self.out_channels, self.T, *HWD)

        return x if y is None else (x, y)


# --------------------------------------------------------------------------------------------------

class _ConvBlock(nn.ModuleDict):
    def __init__(
            self,
            f_config,
            activ_func=None,
            conv_kernel_size=3,
            dropout_rate=0.,
            norm_func=None,
            use_residuals=False,
            X=3,
            T=2
    ):
        super(_ConvBlock, self).__init__()
        self.n_layers = len(f_config) - 1
        self.use_residuals = use_residuals

        self.layers = nn.ModuleList([
            _ConvLayer(
                n_input_features=f_config[n],
                n_output_features=f_config[n + 1],
                activ_func=activ_func,
                conv_kernel_size=conv_kernel_size,
                dropout_rate=dropout_rate,
                norm_func=norm_func,
                X=X,
                T=T
            ) for n in range(self.n_layers)]
        )

    def forward(self, x):
        for n, layer in enumerate(self.layers):
            if self.use_residuals and layer.conv.in_channels == layer.conv.out_channels:
                res = x
                x = layer(x) + res
            else:
                x = layer(x)
        return x


class _ConvLayer(nn.Module):
    def __init__(
            self,
            n_input_features,
            n_output_features,
            activ_func=None,
            conv_kernel_size=3,
            dropout_rate=0.,
            norm_func=None,
            X=3,
            T=2
    ):
        super(_ConvLayer, self).__init__()
        conv_kernel_size = (
            [conv_kernel_size] * X if isinstance(conv_kernel_size, int) else conv_kernel_size
        )
        conv_padding_size = [
            ((cs - 1) // 2) if cs % 2 == 1 else (cs // 2) for cs in conv_kernel_size
        ]
        self.T = T
        self.X = X

        self.conv = eval(f'nn.Conv{X}d')(
            in_channels=(n_input_features),
            out_channels=(n_output_features),
            kernel_size=conv_kernel_size[-X:],
            padding=conv_padding_size[-X:],
            bias=False if norm_func is not None else True
        )

        self.layer_funcs = nn.Sequential(
            norm_func(n_output_features) if norm_func is not None else nn.Identity(),
            activ_func() if activ_func is not None else nn.Identity(),
            eval(f'nn.Dropout{X}d')(p=dropout_rate) if dropout_rate > 0. else nn.Identity()
        )

    def forward(self, x):
        B = x.shape[0]
        HWD = x.shape[-self.X:]

        x = self.layer_funcs(self.conv(x))
        return x


class _Flatten(nn.Module):
    def __init__(
            self,
            n_features,
            kernel_size,
            pool_type='Avg',
            T=2,
            X=3
    ):
        super(_Flatten, self).__init__()
        self.X = X
        self.T = T

        self.flatten = eval(f'nn.Adaptive{pool_type}Pool{X}d')((1,) * self.X)

    def forward(self, x):
        x = self.flatten(x).view(*x.shape[:-self.X]).squeeze(dim=-1)
        return x


class _LinearBlock(nn.Module):
    def __init__(
            self,
            f_config,
            activ_func=None,
            conv_size=3
    ):
        super(_LinearBlock, self).__init__()
        self.n_layers = len(f_config) - 1
        padding_size = ((conv_size - 1) // 2 if conv_size % 2 == 1 else conv_size // 2)

        # Add intermediate layers
        for n in range(self.n_layers - 1):
            self.add_module(
                f'LinearLayer{n+1}', nn.Sequential(
                    nn.Linear(in_features=f_config[n], out_features=f_config[n + 1]),
                    activ_func() if activ_func is not None else nn.Identity()
                )
            )

        # Add last linear function
        self.LastLinear = nn.Linear(
            in_features=f_config[-2], out_features=f_config[-1],
        )

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, list) else x

        for n in range(self.n_layers - 1):
            x = self.__getattr__(f'LinearLayer{n+1}')(x)
        x = self.LastLinear(x)

        return x


class _MLPBlock(nn.ModuleDict):
    def __init__(
            self,
            f_config,
            activ_func=None,
            norm_func=None,
            do_softmax=True,
    ):
        super(_MLPBlock, self).__init__()
        self.n_layers = len(f_config) - 1

        self.layers = nn.ModuleList([
            _MLPLayer(
                n_input_features=f_config[n],
                n_output_features=f_config[n + 1],
                activ_func=(nn.Softmax(dim=-1) if n == (self.n_layers - 1) and do_softmax
                            else activ_func() if activ_func is not None and n < (self.n_layers - 1)
                            else None),
                norm_func=(norm_func if n < (self.n_layers - 1) else None)
            ) for n in range(self.n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:            
            x = layer(x)
        return x


class _MLPLayer(nn.Module):
    def __init__(
            self,
            n_input_features,
            n_output_features,
            activ_func=None,
            norm_func=None,
    ):
        super(_MLPLayer, self).__init__()

        self.linear = nn.Linear(
            in_features=n_input_features,
            out_features=n_output_features,
            bias=False if norm_func is not None else True
        )
        """
        self.layer_funcs = nn.Sequential(
            norm_func(n_output_features) if norm_func is not None else nn.Identity(),
            activ_func() if activ_func is not None else nn.Identity(),
        )
        """
        # self.norm = norm_func(n_output_features) if norm_func is not None else nn.Identity()
        self.activ = (activ_func if activ_func is not None else nn.Identity())

    def forward(self, x):
        x = self.linear(x)
        x = self.activ(x)
        return x


class _Pool(nn.Module):
    def __init__(
            self,
            pool_type='Max',
            kernel_size=2,
            out_shape=(1, 1, 1),
            X=3,
            T=2,
            **kwargs
    ):
        super(_Pool, self).__init__()
        self.T = T
        self.X = X

        if pool_type == 'AdaptiveAvg':
            self.pool = eval(f'nn.{pool_type}Pool{X}d')(out_shape)
        else:
            self.pool = eval(f'nn.{pool_type}Pool{X}d')(
                kernel_size=kernel_size,
                stride=kernel_size,
            )

    def forward(self, x):
        return self.pool(x)


class _UpConv(nn.Module):
    def __init__(
            self,
            n_features,
            kernel_size=2,
            X=3,
            T=2
    ):

        super(_UpConv, self).__init__()
        self.T = T
        self.X = X

        self.upsample = eval(f'nn.ConvTranspose{X}d')(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True
        )

    def forward(self, x, siz):
        return self.upsample(x)


# --------------------------------------------------------------------------------------------------

def _parse_arg_as_list(cls, arg, dtype):
    varname = f'{arg=}'.split('=')[0]

    if not isinstance(arg, list):
        arg = list(arg) if isinstance(arg, tuple) else [arg] * (cls.X + 1)

    if len(arg) != cls.X + 1:
        if len(arg) == 1:
            arg = arg * (cls.X + 1)
        else:
            varname = f'{arg=}'.split('=')[0]
            utils.arg_error(
                f'{varname} must be {dtype} or list/tuple of {dtype} of '
                f'len=={cls.X+1} (input was {arg})', cls
            )

    return arg


def _parse_arg_as_function(cls, arg):
    possible_funcs = [f'{arg}', f'nn.{arg}', f'{arg}{cls.X}d', f'nn.{arg}{cls.X}d']
    found = False

    for func in possible_funcs:
        try:
            arg = eval(func)
            found = True
        except (AttributeError, NameError):
            pass

    if not found:
        varname = f'{arg=}'.split('=')[0]
        utils.arg_error(
            f'{arg} (or {arg}{cls.X}d) is not a valid input for {varname} (not a valid callable '
            f'function or an attribute of torch.nn)', cls
        )
    return arg
