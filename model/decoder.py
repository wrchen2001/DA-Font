from functools import partial
import torch
import torch.nn as nn
from .modules import ConvBlock, ResBlock


class Integrator(nn.Module):
    """
    Integrator module for combining component, content, and optional reference features
    """

    def __init__(self, C, norm='none', activ='none', C_content=0, C_reference=0):
        super().__init__()
        C_in = C + C_content + C_reference  # Total input channels = component + content + reference channels
        self.integrate_layer = ConvBlock(C_in, C, 1, 1, 0, norm=norm, activ=activ)

    def forward(self, comps, content=None, reference=None):
        """
        Args:
            comps (Tensor): [B, C, H, W] component features
            content (Tensor): [B, C_content, H, W] content features
            reference (Tensor, optional): [B, C_reference, H, W] reference features

        Returns:
            Tensor: Integrated features [B, C, H, W]
        """
        if reference is None:
            inputs = torch.cat((comps, content), dim=1)
            out = self.integrate_layer(inputs)
            return out
        else:
            inputs = torch.cat((comps, content, reference), dim=1)
            out = self.integrate_layer(inputs)
            return out


class Decoder(nn.Module):
    """
    Image decoder consisting of ResBlocks and ConvBlocks with upsampling
    """

    def __init__(self, layers, skips=None, out='sigmoid'):
        super().__init__()
        self.layers = nn.ModuleList(layers)

        if out == 'sigmoid':
            self.out = nn.Sigmoid()
        elif out == 'tanh':
            self.out = nn.Tanh()
        else:
            raise ValueError(out)

    def forward(self, x):
        """
        Forward pass of decoder

        Args:
            x (Tensor): Input features [B, C, H, W]

        Returns:
            Tensor: Reconstructed output after activation
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return self.out(x)


def dec_builder(C, C_out, norm='none', activ='relu', out='sigmoid'):
    """
    Build Decoder with ResBlocks and upsampling ConvBlocks

    Args:
        C (int): Base channel multiplier
        C_out (int): Number of output channels
        norm (str): Normalization type
        activ (str): Activation function
        out (str): Output activation ('sigmoid' or 'tanh')

    Returns:
        Decoder: Constructed decoder module
    """
    ConvBlk = partial(ConvBlock, norm=norm, activ=activ)
    ResBlk = partial(ResBlock, norm=norm, activ=activ)

    layers = [
        ResBlk(C * 8, C * 8, 3, 1, norm="in"),
        ResBlk(C * 8, C * 8, 3, 1, norm="in"),
        ResBlk(C * 8, C * 8, 3, 1, norm="in"),
        ConvBlk(C * 8, C * 4, 3, 1, 1, norm="in", upsample=True),
        ConvBlk(C * 4, C * 2, 3, 1, 1, norm="in", upsample=True),
        ConvBlk(C * 2, C * 1, 3, 1, 1, norm="in", upsample=True),
        ConvBlk(C * 1, C_out, 3, 1, 1, norm="in")
    ]

    return Decoder(layers, out=out)



