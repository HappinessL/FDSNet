import torch
import torch.nn as nn
from functools import partial
from torch.autograd import Variable
import torch.nn.functional as F
import cv2

from DWT import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                     padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                     padding=1, bias=True)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out        

# Transformer model
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads  
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.headconv = nn.Conv2d(num_heads, num_heads, kernel_size=3, padding=1, stride=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b, c, h, w = x.shape  
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)  
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)  
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.headconv(attn)
        attn = attn.softmax(dim=-1)
        out = (attn @ v)  
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class SAAttn(nn.Module):
    def __init__(self, in_channels, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        self.act_fn = act_fn()
        self.gate_fn = gate_fn()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.act_fn(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2) # [B, C', H, W]
        ori_x = x
        x = self.conv(x)
        attn = self.gate_fn(x)  # [B, 1, H, W]
        x = ori_x * attn
        x = x.flatten(2).transpose(1, 2) # [B, N, C]
        return x

class AttnMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.attn = SAAttn(out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.attn(x, H, W)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, bias=False, mlp_ratio=2., drop_path=0., qkv_bias=False, qk_scale=None, dropout_rate=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = AttnMlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim)
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)
        x = self.norm1(x).transpose(1, 2).reshape(b, c, h, w)
        x = x + self.attn(x)
        x = x.reshape(b, c, h*w).transpose(1, 2)
        x = x + self.mlp(self.norm2(x), h, w)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, gc=64, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

class FrequencySplitModule(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, channels=32, block_num=[]):
        super(FrequencySplitModule, self).__init__()
        self.down1 = HaarDownsampling(channel_in)
        current_channel = channel_in * 4
        self.up = HaarDownsampling(channel_in)
        self.split_len1 = channel_out                        
        self.split_len2 = current_channel - self.split_len1  
        self.conv_LF_first = nn.Conv2d(self.split_len1, channels, 3, 1, 1)
        self.conv_LF_last = nn.Conv2d(channels, self.split_len1, 3, 1, 1)
        self.conv_HF_first = nn.Conv2d(self.split_len2, channels, 3, 1, 1)
        self.conv_HF_last = nn.Conv2d(channels, self.split_len2, 3, 1, 1)

        LFBlock = []
        for i in range(block_num[1]):
            LFBlock.append(TransformerBlock(dim=channels, num_heads=2))
        self.LFBlock = nn.Sequential(*LFBlock)

        HFBlock = []
        for i in range(block_num[0]):
            HFBlock.append(DenseBlock(channels, channels))
        self.HFBlock = nn.Sequential(*HFBlock)
        self.attention = FrequencyAttentionModule(kernel_size=3)
        
    def forward(self, x, rev=None, cal_jacobian=False):
        jacobian = 0
        B, C, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode='bicubic')  
        x = self.down1(x, rev=False)
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        x1 = self.conv_LF_first(x1)
        x1 = self.LFBlock(x1)
        x1 = self.conv_LF_last(x1) 
        x2 = self.conv_HF_first(x2)
        x2 = self.HFBlock(x2)
        x2 = self.conv_HF_last(x2)  
        x = torch.cat((x1, x2), 1)  

        att = self.attention(x)
        x2 = att * x2
        x = torch.cat((x1, x2), 1)  

        x = self.up1(x, rev=True) 
        return x

class UpNet(nn.Module):
    def __init__(self, in_chans=3, n_stage=2, block_num=(4, 4), upsampler='pixelshuffle',
                 upscale=4, num_feat=64, img_range=1., resi_connection='3conv', **kwargs):
        super(UpNet, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        features = []
        for i in range(n_stage):
            features.append(FrequencySplitModule(channel_in=num_in_ch, channel_out=num_out_ch, block_num=block_num))
        self.Features = nn.Sequential(*features)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        
        x = self.Features(x)
        
        x = x / self.img_range + self.mean
        return x

