"""
Implementation of CLIP model
Adapted from: https://github.com/openai/CLIP/blob/main/clip/clip.py
"""

import os
import urllib
import hashlib
import warnings
import math
import torch
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from collections import OrderedDict
from typing import Tuple, Union
from .utils import log_info
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
import numpy as np 
import json 

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
}

_PT_NAME = {
    "RN50": "RN50.pt",
    "RN101": "RN101.pt",
    "RN50x4": "RN50x4.pt",
    "RN50x16": "RN50x16.pt",
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
    "ViT-L/14": "ViT-L-14.pt"
}



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Adapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, T):
        BT, L, C = x.size() 
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x_id = x
        x = x[:, 1:, :]
        x = self.fc1(x)
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()

        x = self.conv(x)

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        x = self.fc2(x)
        x_id[:, 1:, :] += x
        return x_id

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, vt_choice:str, attn_mask=None, block_id=1, args=None):
        """
        Args:
            block_id: the id the the block in the whole model, start from 1
        """
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.args = args 

        if vt_choice == 'v' and block_id<=args.num_adapter_layers:
            self.temp_adapter = Adapter(in_channels=d_model, adapter_channels=args.adapter_channels, kernel_size=(args.temp_kernel_sz,1,1))
        else:
            self.temp_adapter = None 

    def attention(self, q: torch.Tensor,k: torch.Tensor, v: torch.Tensor, mask=None, need_weights=False):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(q.size(0)) 

        attn_mask_ = attn_mask_.to(dtype=q.dtype, device=q.device) if attn_mask_ is not None else None
        if mask is not None:
            attn_mask_ = mask.to(dtype=q.dtype, device=q.device)
        
        if need_weights is False:
            output = self.attn(q, k, v, need_weights=False, attn_mask=attn_mask_)[0]
            return output
        else:
            output, weights = self.attn(q, k, v, need_weights=True, attn_mask=attn_mask_)
            return output, weights

    def forward(self, x, **kwargs):

        if  'visual' in kwargs:  
            # x: [Ne+Nvp*n1+(G*G+1)*Nf, Nv, Ev]
            B,  dim = x.shape[1], x.shape[-1]
            Nvp, n1, Nf, Ne = self.args.Nvp, self.args.n1, self.args.Nf, self.args.Ne
            visual_prompt,frame_token= x[:Ne,:,:],x[Ne:,:,:] 

            frame_token = self.ln_1(frame_token) # [Nvp*n1+(G*G+1)*Nf, Nv, Ev]
            visual_prompt = self.ln_1(visual_prompt) # [Ne, Nv, Ev]

            all_prompt = frame_token[:Nvp*n1, :, :] # [Nvp*n1, Nv, Ev]
            all_cls_patch = frame_token[Nvp*n1:, :, :].reshape(-1, Nf*B, dim) # [G*G+1, Nf*Nv, Ev]
            
            attention_output_frames = self.attention(all_cls_patch, all_cls_patch, all_cls_patch) # [G*G+1, Nf*Nv, Ev]

            attention_output_frames = attention_output_frames.reshape(-1, Nf, B, dim).permute(2, 1, 0, 3).flatten(0,1) # [Nv*Nf, G*G+1, Ev]
            attention_output_frames = self.temp_adapter(attention_output_frames, Nf)
            attention_output_frames = attention_output_frames.reshape(B, Nf, -1, dim).permute(2, 1, 0, 3) # [G*G+1, Nf, Nv, Ev]
            attention_output_frames = attention_output_frames.reshape(-1, B, dim) # [G*G+1, Nf*Nv, Ev]


            event_query = visual_prompt # [Ne, Nv, Ev]
            # NOTE: features processed with temporal adapters are used for the next layer
            event_kv = torch.cat((visual_prompt,frame_token.reshape(-1,B,dim)),dim=0).to(x.device) # [Ne+Nvp*n1+(G*G+1)*Nf, Nv, Ev]

            if 'visual_attn_mask' in kwargs:
                visual_attn_mask = kwargs['visual_attn_mask'] # [Ne, Ne+Nvp*n1+(G*G+1)*Nf]
            else:
                visual_attn_mask = None 
            

            attention_output_prompt, weights = self.attention(event_query,event_kv,event_kv,mask=visual_attn_mask,need_weights=True) 
            x = x + torch.cat((attention_output_prompt,all_prompt,attention_output_frames),dim=0) 
            x = x + self.mlp(self.ln_2(x))

            return (x, weights) # [Ne+Nvp*n1+(G*G+1)*Nf, Nv, Ev], [Nv, Ne, Ne+Nvp*n1+(G*G+1)*Nf]
        else:
            # x: [Nt, Lt, Et]
            L = x.shape[0]
            text_prompt = kwargs['text_prompt'].permute(1, 0, 2) # [Ntp, Lt, Et]
            Lp = text_prompt.shape[0] // 2 

            x_with_prompt = torch.cat((
                x[:1,:, :], # [SOS]
                text_prompt[:Lp,:,:], # prefix 
                x[1:,:,:], 
                text_prompt[Lp:,:,:]  # postfix 
            ), dim=0) # [Nt+Ntp, Lt, Et]


            x_with_prompt_ln = self.ln_1(x_with_prompt)
            x_with_prompt = x_with_prompt + self.attention(x_with_prompt_ln,x_with_prompt_ln,x_with_prompt_ln)
            x_with_prompt = x_with_prompt + self.mlp(self.ln_2(x_with_prompt))
            x = torch.cat((
                x_with_prompt[:1, :, :],
                x_with_prompt[Lp+1:Lp+L, :, :]
            )) # [Nt, Lt, Et]
            return (x, )


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, vt_choice: str, attn_mask = None, args=None):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, vt_choice, attn_mask, i + 1,args)
                                            for i in range(layers)])

    def forward(self, x: torch.Tensor, **kwargs):
        return self.resblocks(x, **kwargs)




class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                     linear_patch: str = '2d',
                    video_frames=None, args=None):
        super().__init__()
        self.args = args 
        self.configure_pyramid()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.width = width
        assert linear_patch in ['2d', '3d']

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width)) 
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2+1 , width))
        
        if args.time_embedding != 0: 
            self.frame_embedding = nn.Parameter(scale * torch.randn(video_frames,width).unsqueeze(1))
        else:
            self.frame_embedding = None
            
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, vt_choice='v', args=args)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.visual_token_len = (input_resolution // patch_size) **2 + 1
        
        ############################################ NEW ADDED CODE ############################################
        # here, global_prompts are actually event prompts
        self.global_prompts = nn.Parameter(torch.zeros( 
            1, self.args.Ne, self.width))

        patch_size = _pair(patch_size)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.width))  # noqa
        nn.init.uniform_(self.global_prompts.data, -val, val)

        self.generate_proposals()
        self.generate_attn_mask() 
    
    def configure_pyramid(self):
        with open(self.args.pyr_config_path, "r") as f:
            pyr_config = json.load(f)
        Nf = pyr_config["Nf"]
        if "H" in pyr_config: # pyr_config/cfg_32.json
            self.H = pyr_config["H"]
        elif "offset" in pyr_config: # pyr_config/cfg_32_01111.json
            # another improved and easier construction method. 
            # 1 for overalapping (offset=seg_len//2) layers, 0 for non-overlapping (offset=seg_len) layers.
            # will be detailed in our journal version
            # pyr_config/cfg_32_01111.json will produce same results as pyr_config/cfg_32.json
            offset = pyr_config["offset"] 
            self.sel_layers = [(0, 1, Nf)] 
            p = int(math.log(Nf, 2))
            assert 2**p == Nf
            for i in range(1, p+1):
                seg_len = 2 ** i
                k = seg_len-1 
                if k == Nf-1:
                    self.sel_layers.append((k, 1, 1))
                else:
                    if offset[i-1] == 1: # overlapping
                        ok0 = seg_len // 2
                    elif offset[i-1] == 0: # non-overlapping
                        ok0 = seg_len
                    else:
                        raise NotImplementedError
                    
                    assert (Nf-seg_len) % ok0 == 0
                    nk = (Nf-seg_len)//ok0+1
                    self.sel_layers.append((k, ok0, nk))
            print("sel_layers: ", self.sel_layers)

            structure = []
            m = len(self.sel_layers)
            for i in range(m-1):
                k2, ok2, nk2 = self.sel_layers[i]
                k1, ok1, nk1 = self.sel_layers[i+1] # k1>k2
                c_k1k2 = (k1-k2)//ok2 + 1
                if nk1 == 1:
                    o_k1k2 = 1
                else:
                    o_k1k2 = (nk2-c_k1k2)//(nk1-1)
                structure.append([c_k1k2, o_k1k2])
            print("structure: ", structure)
            self.H = structure
        else:
            raise NotImplementedError
        
        self.num_prompt_layers = len(self.H)
        self.kernel_and_offset = {}
        self.cnts = [Nf] # raw frames (0-th layer)
        for k in range(self.num_prompt_layers, 0, -1):
            self.kernel_and_offset[(k, k-1)] = self.H[k-1]
        
            
        for k in range(self.num_prompt_layers):
            n_prev = self.cnts[-1]
            c, o = self.H[k]
            assert (n_prev - c) % o == 0
            n = (n_prev - c) // o + 1
            self.cnts.append(n)
        
        self.level_event_cnts = self.cnts[1:][::-1] # top first
        

        for k1 in range(self.num_prompt_layers, 0, -1): 
            for k2 in range(k1-1, -1, -1):
                if k2==k1-1:
                    continue
                else:
                    o_prev = self.kernel_and_offset[(k1, k2+1)][1]
                    o_k1k2 = o_prev * self.kernel_and_offset[(k2+1, k2)][1]
                    c_k1k2 = self.cnts[k2] - o_k1k2 * (self.cnts[k1] - 1)
                    self.kernel_and_offset[(k1, k2)] = [c_k1k2, o_k1k2]
        
        # update args
        self.args.Nf = Nf 
        self.args.Ne = sum(self.level_event_cnts)
        self.args.n1 = self.level_event_cnts[-1]
        self.args.Le = len(self.level_event_cnts)

    
    def generate_proposals(self): # for vcmr
        proposals = []
        for l_i, n_i in enumerate(self.level_event_cnts):
            k, s = self.kernel_and_offset[(self.num_prompt_layers-l_i, 0)]
            for i in range(n_i):
                st_i = (i * s) / self.args.Nf
                ed_i = (i * s + k) / self.args.Nf 
                proposals.append([st_i, ed_i])
        self.proposals = torch.tensor(proposals)
                
    
    def generate_attn_mask(self):
        Nf = self.args.Nf 
        n1 = self.level_event_cnts[-1]
        Ne = sum(self.level_event_cnts)
        M_size = Ne + Nf 
        Nvp = self.args.Nvp 
        visual_token_len = self.visual_token_len 

        M = torch.full((M_size, M_size), float('-inf'))
        K = self.num_prompt_layers
        for k1 in range(K, -1, -1):
            for k2 in range(k1, -1, -1): 
                st_1, ed_1 = sum(self.cnts[k1+1:]), sum(self.cnts[k1:])
                st_2, ed_2 = sum(self.cnts[k2+1:]), sum(self.cnts[k2:])
                
                if k1 == k2:
                    M[st_1:ed_1, st_2:ed_2].fill_diagonal_(0.)
                else:
                    n_i, n_j = self.cnts[k1], self.cnts[k2]
                    k_ij, s_ij = self.kernel_and_offset[(k1, k2)]
                    for i in range(n_i):
                        M[st_1+i, st_2+i*s_ij:st_2+i*s_ij+k_ij] = 0. 
                        M[st_2+i*s_ij:st_2+i*s_ij+k_ij,st_1+i] = 0.

        self.attn_gg = M[:Ne, :Ne] # [Ne, Ne]
        self.attn_gf = M[:Ne, Ne-n1: Ne].unsqueeze(1).repeat(1, Nvp, 1).reshape(Ne, -1) # [Ne, Nvp*n1]
        self.attn_gp = M[:Ne, Ne:].unsqueeze(1).repeat(1, visual_token_len, 1).reshape(Ne, -1) # [Ne, (G*G+1)*Nf]
        self.visual_attn_mask = torch.cat((self.attn_gg, self.attn_gf, self.attn_gp), dim=1)
        return self.visual_attn_mask



        
    def incorporate_prompt(self, x, frame_prompts):
        # x: [Nv*Nf, G*G+1, Ev], frame_prompts: [Nv, Nvp, Ev]
        BT = x.shape[0]
        B = BT // self.args.Nf
        frame_prompts = frame_prompts.view(B,self.args.Nvp,x.size(-1)).unsqueeze(1).expand(-1,self.args.n1,-1,-1) # [Nv, n1, Nvp, Ev]
        frame_prompts = frame_prompts.permute(0, 2, 1, 3).reshape(B, -1, x.size(-1)) # [Nv, Nvp*n1, Ev]


        x = x.view(B,self.args.Nf,x.size(-2),x.size(-1)) # [Nv, Nf, G*G+1, Ev]
        x = x.permute(0, 2, 1, 3).reshape(B, -1, x.size(-1))  # [Nv, (G*G+1)*Nf, Ev]
        
        x_local_prompt = torch.cat((
            frame_prompts,
            x
        ), dim=1) # [Nv, Nvp*n1+(G*G+1)*Nf, Ev]

        event_prompts = self.global_prompts.expand(B, -1, -1) # [Nv, Ne, Ev]
        
        x_prompt = torch.cat((event_prompts,x_local_prompt),dim=1) # [Nv, Ne+Nvp*n1+(G*G+1)*Nf, Ev]
        
        return x_prompt

    def forward_deep_prompt(self, x,total_frame_prompts, visual_attn_mask=None, return_weights=False):
        hidden_states = None
        B = x.shape[1]

        num_layers = self.transformer.layers

        if return_weights:
            all_weights = [] 

        for i in range(num_layers):
            if i == 0:
                if return_weights:
                    hidden_states, weights = self.transformer.resblocks[i](x, return_weights=True, visual=True, visual_attn_mask=visual_attn_mask)
                    all_weights.append(weights)
                else:
                    hidden_states = self.transformer.resblocks[i](x, visual=True, visual_attn_mask=visual_attn_mask)[0]
            else:
                if i <= len(total_frame_prompts):
                    frame_prompts = total_frame_prompts[i].view(B,self.args.Nvp,x.size(-1)).unsqueeze(1).expand(-1,self.args.n1,-1,-1).permute(2,1,0,3)  
                    frame_prompts = frame_prompts.reshape(-1, B, x.size(-1)) 
                    hidden_states_global = hidden_states[:self.args.Ne, :, :] 
                    hidden_states_frame = hidden_states[self.args.Ne+self.args.Nvp*self.args.n1:, :, :] 
                    hidden_states = torch.cat((hidden_states_global,frame_prompts,hidden_states_frame),dim=0)

                if return_weights:
                    hidden_states, weights = self.transformer.resblocks[i](hidden_states, return_weights=True, visual=True, visual_attn_mask=visual_attn_mask)
                    all_weights.append(weights)
                else:
                    hidden_states = self.transformer.resblocks[i](hidden_states, visual=True, visual_attn_mask=visual_attn_mask)[0]

        if return_weights:
            weights = torch.stack(all_weights, dim=0) # [Nlayers, B, Ne, ...]
            return hidden_states, weights 
        else:
            return hidden_states 

    def forward(self, x, total_frame_prompts, return_weights=False):
        x = self.conv1(x) 
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) 

        x = torch.cat([self.class_embedding.to(x.dtype) + \
                        torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
        
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x) 
        
        x = self.incorporate_prompt(x,total_frame_prompts[0]) 
        x = x.permute(1, 0, 2)  
        if return_weights:
            x, weights = self.forward_deep_prompt(x,total_frame_prompts,visual_attn_mask=self.visual_attn_mask.to(x.device), return_weights=True)
        else:				
            x= self.forward_deep_prompt(x,total_frame_prompts,visual_attn_mask=self.visual_attn_mask.to(x.device))
        x = x.permute(1, 0, 2)  					

        if return_weights:
            return x, weights
        else:
            return x


class CLIP(nn.Module):
    def __init__(self,
                    embed_dim: int,
                    # vision
                    image_resolution: int,
                    vision_layers: Union[Tuple[int, int, int, int], int],
                    vision_width: int,
                    vision_patch_size: int,
                    # text
                    context_length: int,
                    vocab_size: int,
                    transformer_width: int,
                    transformer_heads: int,
                    transformer_layers: int,
                    # vision linear of patch
                    linear_patch: str = '2d',
                    video_frames=None,
                    args=None
                    ):

        super().__init__()
        self.args = args 
        self.context_length = context_length
        # visual encoder
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:  
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch=linear_patch,
                video_frames=video_frames,
                args=args
            )
        # text encoder
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask,
            vt_choice='t',
            args=args
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))

        #### NEW ADDED CODES #####
        
        self.Ep = self.Epv = vision_width
        self.Ept = transformer_width
        self.visual_prompt_embedding = torch.nn.Parameter(torch.zeros(
            1, args.Nvp,args.Nlayers*self.Ep))

        self.prefix_proj = nn.Linear(self.Epv, self.Ept)
        self.postfix_proj = nn.Linear(self.Epv, self.Ept)

        patch_size = _pair(vision_patch_size)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.Epv))  
        nn.init.uniform_(self.visual_prompt_embedding.data, -val, val)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)


        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self, context_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    
    def encode_prompt(self,batch_size, device):
        B = batch_size
        visual_prompt_embedding = self.visual_prompt_embedding.expand(B, -1,-1).to(device)
        visual_prompt = visual_prompt_embedding.view(B,self.args.Nvp,self.args.Nlayers,self.Epv).permute(2,0,1,3)
        text_prefix_prompt = self.prefix_proj(visual_prompt) # V->T
        text_postfix_prompt = self.postfix_proj(visual_prompt)
        unified_text_prompt = torch.cat((text_prefix_prompt,text_postfix_prompt),dim=2) # [Nlayers, Nv, Ntp=2*Npv, Et]
        unified_visual_prompt = visual_prompt # [Nlayers, Nv, Npv, Ev]
        return unified_text_prompt, unified_visual_prompt

    def encode_image(self, image, total_frame_prompts, return_weights=False):
        # image: [Nv*Nf, C, H, W], total_frame_prompts: [Nlayers, Nv, Npv, Ev]
        if return_weights:
            hidden, weights = self.visual(image.type(self.dtype),total_frame_prompts, return_weights=True)
        else: 
            hidden = self.visual(image.type(self.dtype),total_frame_prompts) 
        hidden = self.visual.ln_post(hidden) @ self.visual.proj # [Nv, Ne+n1*Npv+Nf*(G**2+1), E]

        if return_weights:
            return hidden[:,:self.args.Ne,:], weights
        else:
            return hidden[:,:self.args.Ne,:] # [Nv, Ne, E]


    def encode_text(self, text, total_text_prompts):  
        # text: [Nt, Lt], total_text_prompts: [Nlayers, Nt, Ntp, Et]
        x = self.token_embedding(text).type(self.dtype)  
        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype) 
        x = x + pos_emd

        x = x.permute(1, 0, 2)  

        for i in range(self.args.Nlayers):
            x = self.transformer.resblocks[i](x, text_prompt=total_text_prompts[i])[0]

        x = x.permute(1, 0, 2)  

        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection 
        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)] # trick: EOS(40407) has the max token id (so we dont need mask anymore)
        return x # [Nt, E]



def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_clip_model(state_dict: dict, convert_fp16=True, linear_patch='2d', cut_top_layer=0,
                        load_state_dict=True, is_eval=True, 
                        video_frames=None,
                        args=None):
    """build a CLIP model
    Args:
        state_dict: the pretrained weights
        convert_fp16: If True, convert applicable model parameters to fp16
        linear_patch: the patch manner of image / video
        cut_top_layer: abandon a few top layers
        cluster: the number of cluster
        args: all the config arguments
    Return:
        A CLIP model, config of CLIP
    """
    clip_config = {}
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    # print info
    log_info("\n config of CLIP:\n"
                "\t embed_dim: {}\n"
                "\t image_resolution: {},\n"
                "\t vision_layers: {},\n"
                "\t vision_width: {},\n"
                "\t vision_patch_size: {},\n"
                "\t video_frames: {},\n"
                "\t context_length: {},\n"
                "\t vocab_size: {},\n"
                "\t transformer_width: {},\n"
                "\t transformer_heads: {},\n"
                "\t transformer_layers: {},\n".format(embed_dim, image_resolution, vision_layers,
                vision_width, vision_patch_size, video_frames,
                context_length, vocab_size, transformer_width,
                transformer_heads, transformer_layers))
    clip_config['context_length'] = context_length
    clip_config['transformer_width'] = transformer_width
    clip_config['transformer_heads'] = transformer_heads

    model = CLIP(
            embed_dim,
            image_resolution, vision_layers - cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads,
            transformer_layers - cut_top_layer,
            linear_patch=linear_patch, video_frames=video_frames, args=args
        ).float()

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    if convert_fp16:
        convert_weights(model)
    
    if load_state_dict:
        model.load_state_dict(state_dict)

    if is_eval:
        model.eval()

    return model, clip_config


##############################################################################
# utils for downloading CLIP pretrained weights and loading the pretrained state_dict
# https://github.com/openai/CLIP/blob/main/clip/clip.py
# 
##############################################################################

def load_clip_state_dict(pretrained_clip_name="ViT-B/32", pretrained_dir=os.path.expanduser("~/models/pretrained")):
    """load pretrained CLIP state dict from local file
    Args:
        pretrained_clip_name: name of pretrained CLIP model
        pretrained_dir: where the pretrained weight file located
    """
    if pretrained_clip_name in _MODELS and pretrained_clip_name in _PT_NAME:
        model_path = os.path.join(pretrained_dir, _PT_NAME[pretrained_clip_name])
    else:
        raise NotImplementedError('Do not find CLIP model with name {}'.format(pretrained_clip_name))

    if pretrained_clip_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14"] and os.path.exists(model_path):
        pass
    else:
        raise IOError("Not found {}".format(model_path))
        if pretrained_clip_name in _MODELS:
            model_path = _download(_MODELS[pretrained_clip_name], root=pretrained_dir)
        elif os.path.isfile(pretrained_clip_name):
            model_path = pretrained_clip_name
        else:
            raise RuntimeError(f"Model {pretrained_clip_name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    return state_dict


def _download(url: str, root: str = os.path.expanduser("~/models/pretrained")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models():
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())
