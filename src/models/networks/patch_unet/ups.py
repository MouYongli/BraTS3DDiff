
#Upsample module

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Conv



class UpTransConvBlock(nn.Sequential):
    "TransConv upsample + Conv layer"
    " transconv: out_size = (in_size - 1) * stride - 2 * padding + kernel_size + output_padding"
    
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        up_kernel_size: Sequence[int] | int,
        up_stride: Sequence[int] | int,
        up_padding: Sequence[int] | int,
        up_output_padding: Sequence[int] | int,
        act: Union[str, tuple] = (
            "LeakyReLU",
            {"negative_slope": 0.1, "inplace": True},
        ),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
        """
        super().__init__()
        
        self.add_module('transconv_0',
            Conv[Conv.CONVTRANS, spatial_dims](
                in_channels=in_chns,
                out_channels=in_chns,
                kernel_size=up_kernel_size,
                stride=up_stride,
                padding=up_padding,
                output_padding=up_output_padding,
                bias=bias,
            )
        )
        self.add_module('conv_0',
            Convolution(
                spatial_dims,
                in_chns,
                out_chns,
                act=act,
                norm=norm,
                dropout=dropout,
                bias=bias,
                padding=1,
            )
        )

class Up6to16(nn.Module):
    """upsamples a in_chns*6*6*6 tensor to a out_chns*16*16*16 tensor using 1 UpTransConvBlock block."""

    def __init__(
        self,
        spatial_dims: int = 3,
        in_chns: int = 1,
        out_chns: int = 1,
        act: Union[str, tuple] = (
            "LeakyReLU",
            {"negative_slope": 0.1, "inplace": True},
        ),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()
        self.in_chns = in_chns
        self.out_chns = out_chns

        self.upsample = UpTransConvBlock(
            spatial_dims,
            in_chns,
            out_chns,
            up_kernel_size=3,
            up_stride=3,
            up_padding=1,
            up_output_padding=0,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        assert x.shape[1:] == (self.in_chns, 6, 6, 6)
        x = self.upsample(x)
        assert x.shape[1:] == (self.out_chns, 16, 16, 16)
        return x


class Up8to32(nn.Module):
    """upsamples a in_chns*8*8*8 tensor to a out_chns*32*32*32 tensor using 1 UpTransConvBlock block."""

    def __init__(
        self,
        spatial_dims: int = 3,
        in_chns: int = 1,
        out_chns: int = 1,
        act: Union[str, tuple] = (
            "LeakyReLU",
            {"negative_slope": 0.1, "inplace": True},
        ),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()
        self.in_chns = in_chns
        self.out_chns = out_chns

        " transconv: 32 = (8 - 1) * stride - 2 * padding + kernel_size + output_padding"

        self.upsample = UpTransConvBlock(
            spatial_dims,
            in_chns,
            out_chns,
            up_kernel_size=4,
            up_stride=4,
            up_padding=0,
            up_output_padding=0,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
        )
    def forward(self, x: torch.Tensor):
        assert x.shape[1:] == (self.in_chns, 8, 8, 8)
        x = self.upsample(x)
        assert x.shape[1:] == (self.out_chns, 32, 32, 32)
        return x
