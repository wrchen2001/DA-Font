import math
import torch
import torch.nn as nn
import torch.nn
import torch.nn.functional as F
from einops import rearrange

from copy import deepcopy


# Define Component Attention Block

class GFP(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.1):
        super(GFP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.attn_fc = nn.Linear(2 * out_features, 1, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features, eps=1e-6)

    def calculate_similarity_matrix(self, features):
        similarity = torch.matmul(features, features.transpose(1, 2))  

        similarity = F.softmax(similarity, dim=-1)
        return similarity

    def forward(self, node_features, adj_matrix):
        """
        Args:
            node_features (Tensor): [B, N, C] input features
            adj_matrix (Tensor or None): optional adjacency matrix
        """

        B, N, C = node_features.shape

        transformed_features = node_features

        attn_input = torch.cat([transformed_features.unsqueeze(2).expand(-1, -1, N, -1), 
                                transformed_features.unsqueeze(1).expand(-1, N, -1, -1)], dim=-1) 
        attn_scores = self.attn_fc(attn_input).squeeze(-1) 

        if adj_matrix is None:
            adj_matrix = self.calculate_similarity_matrix(transformed_features)  

        attn_scores = attn_scores + adj_matrix

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        out = torch.matmul(attn_weights, transformed_features) 
        out = self.layer_norm(out + node_features)

        return out
    

class ComponentAttentionBlock(nn.Module):
    def __init__(self, num_heads=8, num_channels=256, dropout=0.1):
        super(ComponentAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels

        self.gat_layer = GFP(num_channels, num_channels, num_heads, dropout)
        
        # Linear layers for key, value, and query
        self.linears_key = nn.Linear(num_channels, num_channels, bias=False)
        self.linears_value = nn.Linear(num_channels, num_channels, bias=False)
        self.linears_query = nn.Linear(num_channels, num_channels, bias=False)

        # Multi-head concatenation and normalization
        self.multihead_concat_fc = nn.Linear(num_channels, num_channels, bias=False)
        self.layer_norm = nn.LayerNorm(num_channels, eps=1e-6)

    def get_kqv_matrix(self, fm, linears):
        # matmul with style featuremaps and content featuremaps
        return linears(fm)


    def get_reference_key_value(self, reference_map_list):
        B, K, C, H, W = reference_map_list.shape
        m = self.num_heads
        d_channel = self.num_channels // self.num_heads
        reference_sequence = rearrange(reference_map_list, 'b k c h w -> b (k h w) c')

        key_reference_matrix = self.get_kqv_matrix(reference_sequence, self.linears_key)  
        key_reference_matrix = torch.reshape(key_reference_matrix, (B, K * H * W, m, d_channel))
        key_reference_matrix = rearrange(key_reference_matrix, 'b khw m d -> b m khw d')

        value_reference_matrix = self.get_kqv_matrix(reference_sequence, self.linears_value)  
        value_reference_matrix = torch.reshape(value_reference_matrix, (B, K * H * W, m, d_channel))
        value_reference_matrix = rearrange(value_reference_matrix, 'b khw m d -> b m khw d')

        return key_reference_matrix, value_reference_matrix


    def get_component_query(self, component_sequence):
        B, N, C = component_sequence.shape
        d_channel = self.num_channels // self.num_heads
        m = self.num_heads
        query_component_matrix = self.get_kqv_matrix(component_sequence, self.linears_query)
        query_component_matrix = torch.reshape(query_component_matrix, (B, N, m, d_channel))
        query_component_matrix = rearrange(query_component_matrix, 'b n m d -> b m d n')

        return query_component_matrix
    
    def cross_attention(self, query, key, value, mask=None, dropout=None):
        residual = query
        query = self.get_component_query(query) 
        scores = torch.matmul(key, query)  
        scores = rearrange(scores, 'b m khw n -> b m n khw')
        p_attn = F.softmax(scores, dim=-1)

        out = torch.matmul(p_attn, value)  

        out = rearrange(out, 'b m n d -> b n (m d)')
        out = self.multihead_concat_fc(out)

        out = self.gat_layer(out, adj_matrix=None) 
        out = self.layer_norm(out + residual)
        return out
    
    def forward(self, component_sequence, reference_map_list):

        key_reference_matrix, value_reference_matrix = self.get_reference_key_value(reference_map_list)

        style_components = self.cross_attention(component_sequence, key_reference_matrix, value_reference_matrix)

        return style_components


# Define Relation Attention Block

# SoftPooling used for local feature enhancement
class SoftPooling1D(torch.nn.Module):
    def __init__(self,kernel_size,stride=None,padding=0):
        super(SoftPooling1D, self).__init__()
        self.avgpool = torch.nn.AvgPool1d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        x_exp = torch.exp(torch.clamp(x, min=-10, max=10))
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool 


class LFR(nn.Module):
    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, f, 1),
            SoftPooling1D(7, stride=3),
            nn.Conv1d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(f, channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )            
    def forward(self, x):
        g = self.gate(x[:,:1].clone())
        w = F.interpolate(self.body(x), (x.size(2), ), mode='linear', align_corners=False)
        return x * w * g 


# Relation Attention Block for feature fusion between content and components
class RelationAttentionBlock(nn.Module):
    def __init__(self, num_heads=8, num_channels=256):
        super(RelationAttentionBlock, self).__init__()
        self.linears_key = nn.Linear(num_channels, num_channels, bias=False)
        self.linears_value = nn.Linear(num_channels, num_channels, bias=False)
        self.linears_query = nn.Linear(num_channels, num_channels, bias=False)
        self.multihead_concat_fc = nn.Linear(num_channels, num_channels, bias=False)
        self.num_heads = num_heads
        self.layer_norm = nn.LayerNorm(num_channels, eps=1e-6)

        self.LFR1 = LFR(channels = 256)
        self.LFR2= LFR(channels = 100)

    def get_kqv_matrix(self, fm, linears):
        ret = linears(fm)
        return ret

    def get_content_query(self, content_feature_map):
        B, C, H, W = content_feature_map.shape
        m = self.num_heads
        d_channel = C // m
        query_component_matrix = rearrange(content_feature_map, 'b c h w -> b (h w) c')
        query_component_matrix = self.get_kqv_matrix(query_component_matrix, self.linears_query)
        query_component_matrix = torch.reshape(query_component_matrix, (B, H * W, m, d_channel))
        query_component_matrix = rearrange(query_component_matrix, 'b hw m d_channel -> (b m) hw d_channel')

        return query_component_matrix

    def get_component_key_value(self, component_sequence, keys=False):
        B, N, C = component_sequence.shape
        m = self.num_heads
        d_channel = C // m

        if keys:
            key_component_matrix = self.get_kqv_matrix(component_sequence, self.linears_key)
            key_component_matrix = torch.reshape(key_component_matrix, (B, N, m, d_channel))
            key_component_matrix = rearrange(key_component_matrix, 'b n m d_channel -> (b m) n d_channel')
            return key_component_matrix
        else:
            value_component_matrix = self.get_kqv_matrix(component_sequence, self.linears_value)
            value_component_matrix = torch.reshape(value_component_matrix, (B, N, m, d_channel))
            value_component_matrix = rearrange(value_component_matrix, 'b n m d_channel -> (b m) n d_channel')
            return value_component_matrix

    def cross_attention(self, content_feature, components, style_components, mask=None, dropout=None):

        B, C, H, W = content_feature.shape
        content_query = self.get_content_query(content_feature)  
        components_key = self.get_component_key_value(components, keys=True)  
        style_components_value = self.get_component_key_value(style_components) 

        content_query = self.LFR1(content_query)

        style_components_value = self.LFR2(style_components_value)

        residual = content_query
        d_k = content_query.size(-1)
        scores = torch.matmul(content_query, components_key.transpose(-2, -1)) / math.sqrt(d_k) 
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        out = torch.matmul(p_attn, style_components_value)
        out = rearrange(out, '(b m) hw c -> b hw (c m)', m=self.num_heads)

        residual = rearrange(residual, '(b m) hw c -> b hw (c m)', m=self.num_heads)

        out = self.layer_norm(out + residual)  
        out = self.multihead_concat_fc(out)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H)

        return out  

    def forward(self, content_feature, components, style_components):

        transfer_feature = self.cross_attention(content_feature, components, style_components)

        return transfer_feature

