import torch.nn as nn
from functools import partial
from model.modules.blocks import ConvBlock


class ContentEncoder(nn.Module):
    """
    ContentEncoder
    Feature encoder consisting of stacked convolutional blocks
    """

    def __init__(self, layers, sigmoid=False):
        super().__init__()
        self.net = nn.Sequential(*layers)
        self.sigmoid = sigmoid

    def forward(self, x):
        out = self.net(x)
        if self.sigmoid:
            out = nn.Sigmoid()(out)
        return out

def content_enc_builder(C_in, C, C_out, norm='none', activ='relu', pad_type='reflect', content_sigmoid=False):
    """
    Build ContentEncoder network

    Args:
        C_in (int): Number of input channels
        C (int): Base channel multiplier
        C_out (int): Number of output channels
        norm (str): Normalization type for ConvBlock
        activ (str): Activation function for ConvBlock
        pad_type (str): Padding type for ConvBlock
        content_sigmoid (bool): Whether to apply Sigmoid at the output

    Returns:
        ContentEncoder: Constructed encoder module
    """
    
    ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)

    layers = [
        ConvBlk(C_in, C, 3, 1, 1, norm='in', activ='relu'),
        ConvBlk(C * 1, C * 2, 3, 2, 1, norm="in"), 
        ConvBlk(C * 2, C * 4, 3, 2, 1, norm="in"),  
        ConvBlk(C * 4, C * 8, 3, 2, 1, norm="in"), 
        ConvBlk(C * 8, C_out, 3, 1, 1, norm="in")
    ]   

    return ContentEncoder(layers, content_sigmoid)
