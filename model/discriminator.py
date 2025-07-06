import torch
import torch.nn as nn
from functools import partial
from .modules import ResBlock, ConvBlock, w_norm_dispatch, activ_dispatch

class ProjectionDiscriminator(nn.Module):
    """
    Multi-task projection discriminator for font and character classification
    """

    def __init__(self, C, n_fonts, n_chars, w_norm='spectral', activ='none'):
        super().__init__()

        self.activ = activ_dispatch(activ)()
        w_norm = w_norm_dispatch(w_norm)
        self.font_emb = w_norm(nn.Embedding(n_fonts, C))    # Font embedding with weight normalization
        self.char_emb = w_norm(nn.Embedding(n_chars, C))    # Character embedding with weight normalization

    
    def forward(self, x, font_indice, char_indice):
        """
        Args:
            x (Tensor): [B, C, H, W] feature maps
            font_indice (Tensor): [B] font class indices
            char_indice (Tensor): [B] character class indices

        Returns:
            list: [font_out, char_out] projection scores for font and character
        """

        x = self.activ(x)  
        font_emb = self.font_emb(font_indice)   
        char_emb = self.char_emb(char_indice)  

        font_out = torch.einsum('bchw,bc->bhw', x, font_emb).unsqueeze(1)
        char_out = torch.einsum('bchw,bc->bhw', x, char_emb).unsqueeze(1)   

        return [font_out, char_out]


class CustomDiscriminator(nn.Module):
    """
    Discriminator combining ResBlocks, Global Average Pooling, and Multi-task Projection Discriminator
    """

    def __init__(self, feats, gap, projD):
        super().__init__()
        self.feats = feats
        self.gap = gap
        self.projD = projD

    def forward(self, x, font_indice, char_indice):
        """
        Args:
            x (Tensor): [B, 1, H, W] input images
            font_indice (Tensor): [B] font class indices
            char_indice (Tensor): [B] character class indices

        Returns:
            tuple: font and character discrimination outputs
        """

        for layer in self.feats:
            x = layer(x)

        x = self.gap(x)  # Global pooled features
        #ret = [x, x]
        ret = self.projD(x, font_indice, char_indice)
        ret = tuple(map(lambda i: i.cuda(), ret))
        return ret



def disc_builder(C, n_fonts, n_chars, activ='relu', gap_activ='relu', w_norm='spectral',
                 res_scale_var=False):
    
    """
    Build discriminator with spectral norm, ResBlocks, and projection classifier

    Args:
        C (int): Base channel multiplier
        n_fonts (int): Number of font classes
        n_chars (int): Number of character classes
        activ (str): Activation for ConvBlocks and ResBlocks
        gap_activ (str): Activation before Global Average Pooling
        w_norm (str): Weight normalization method
        res_scale_var (bool): If True, use scaled variance in ResBlocks

    Returns:
        CustomDiscriminator: Discriminator model
    """

    ConvBlk = partial(ConvBlock, w_norm=w_norm, activ=activ)
    ResBlk = partial(ResBlock, w_norm=w_norm, activ=activ, scale_var=res_scale_var)

    feats = [
        ConvBlk(1, C, stride=2, activ='none'),   # Initial downsampling
        ResBlk(C * 1, C * 2, downsample=True),  
        ResBlk(C * 2, C * 4, downsample=True),  
        ResBlk(C * 4, C * 8, downsample=True),  
        ResBlk(C * 8, C * 16, downsample=False),  
        ResBlk(C * 16, C * 16, downsample=False), 
    ] 

    gap_activ = activ_dispatch(gap_activ)
    gaps = [
        gap_activ(),
        nn.AdaptiveAvgPool2d(1)  # Global average pooling
    ]   

    projD_C_in = feats[-1].C_out  # Input channels for projection head
    feats = nn.ModuleList(feats)
    gap = nn.Sequential(*gaps)
    projD = ProjectionDiscriminator(projD_C_in, n_fonts, n_chars, w_norm=w_norm)

    disc = CustomDiscriminator(feats, gap, projD)

    return disc
