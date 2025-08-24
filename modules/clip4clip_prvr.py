# coding=utf-8
from __future__ import absolute_import, division, print_function
import os 
import logging
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from .losses import CrossEn
from .base import PreTrainedModel
from .module_cross import CrossConfig, CrossModel
from .utils import all_gather, log_info, update_attr
import importlib
import random
from .clip_propy import build_clip_model
from .clip_propy import load_clip_state_dict
from .module_cross import Transformer as TransformerClip
##add for efficient prompt
from collections import OrderedDict
import copy

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):
        """"create a model from pretrained config or model weights files"""
        # import pdb; pdb.set_trace() 
        task_config = kwargs['task_config']
        if state_dict is None: state_dict = {}

        # load pretrained CLIP state_dict
        pretrained_clip_name = getattr(task_config, 'pretrained_clip_name', "ViT-B/32") 
        clip_state_dict = load_clip_state_dict(pretrained_clip_name, pretrained_dir=task_config.pretrained_dir)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key  
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        # print('cross_model_name',cross_model_name)
        # print('cache_dir',cache_dir)
        # print('type_vocab_size',type_vocab_size)
        # print('state_dict',state_dict)
        # print('task_config',task_config)
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size,
                                                    state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs) 

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)
        # divide the pretrained temperature
        # model.clip.logit_scale.data.fill_(task_config.temperature_new)
        if task_config.temperature_new > 1.0:
            logging.info("Assign new temperature {} to the logit_scale".format(task_config.temperature_new))
            model.clip.logit_scale.data.fill_(task_config.temperature_new)
        
        return model


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class TemporalModelling(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float, attn_mask: torch.Tensor = None, ):
        super(TemporalModelling, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, dropout, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))


class CLIP4Clip_PRVR(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        """
        A CLIP4Clip model in https://github.com/ArrowLuo/CLIP4Clip
        Args:
            cross_config: config arguments of cross_model
            clip_state_dcit: the weight state_dict of pretrained clip
            task_config: config args
        """
        super(CLIP4Clip_PRVR, self).__init__(cross_config)
        # import pdb; pdb.set_trace() 
        self.task_config = task_config
        self.ignore_video_index = -1
        # assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings
        self._stage_one = True
        self._stage_two = False
        # tightTransf or not
        self.loose_type = True if (self._stage_one and task_config.loose_type) else False 
        # patch manner of images / video
        self.linear_patch = getattr(task_config, "linear_patch", '2d') 
        self.sim_header = getattr(task_config, "sim_header", 'meanP')
        if self.sim_header == "tightTransf": assert self.loose_type is False  

        self.video_frames = getattr(task_config, "max_frames", None) 
        self.time_embedding = getattr(task_config, "time_embedding", None) 
        self.freeze_clip = getattr(task_config, "freeze_clip", 0) 
        self.new_added_modules = getattr(task_config, "new_added_modules", [None, ]) 
        self.pre_visual_pooling = getattr(task_config, "pre_visual_pooling", 0)
        # https://github.com/starmemda/CAMoE/blob/main/DSL.py
        self.camoe_dsl = getattr(task_config, "camoe_dsl", False) 
        ##add for efficient prompt
        self.tfm_layers = getattr(task_config,'tfm_layers',None) 
        self.tfm_heads = getattr(task_config,'tfm_heads',None)
        self.dropout = 0.0
        log_info("\n config of CLIP4Clip:\n"
                    "\t Stage-One:{}\n" 
                    "\t Stage-Two:{}\n" 
                    "\t loose type {}\n" 
                    "\t linear_patch: {}\n" 
                    "\t sim_header: {}\n" 
                    "\t camoe_dsl: {}\n".format(self._stage_one, self._stage_two, self.loose_type,
                                                self.linear_patch, self.sim_header, self.camoe_dsl))
        # create CLIP Encoders
        self.clip, clip_config = build_clip_model(clip_state_dict, convert_fp16=True,
                                                    linear_patch=self.linear_patch,
                                                    cut_top_layer=0, load_state_dict=False,
                                                    is_eval=False,
                                                    video_frames=self.video_frames,
                                                    args=task_config)

        cross_config.max_position_embeddings = clip_config['context_length']
        self.loss_fct = CrossEn()


    def forward(self, batch, **kwargs):
        """
            Inputs:
                batch: 
                    'video_names': List[str], len==Nv
                    'video': Tensor, shape: [Nv, Nf, C, H, W]
                    'text_name': List[str], len==Nt
                    'text_ids': Tensor, shape: [Nt, Lt]
                    'text_labels': Tensor, shape: Nt  (text_ids[i] corresponds to video[text_labels[i]])
            Outputs:
                output_dict: 
                    'sequence_output': [Nt, 1, E]
                    'visual_output': [Nv, Ne, E]
                    'loss', 't2v_loss', 'v2t_loss', 'sim_loss': Tensor
        """
        video = batch['video']  
        text_ids = batch['text_ids'] 
        text_labels = batch['text_labels'] 

        v2t_dict = {}
        if type(text_labels) is torch.Tensor:
            text_labels_ = text_labels.tolist() 

        for tid, vid in enumerate(text_labels_):
            if vid not in v2t_dict:
                v2t_dict[vid] = [tid]
            else:
                v2t_dict[vid].append(tid)

        output_dict = {'sequence_output': None,'visual_output': None,'loss': None}
        
        if text_ids is not None and video is not None: 
            batch_size = video.shape[0] 
            # unified_text_prompt: [Nlayers, Nv, Ntp, Et], unified_visual_prompt: [Nlayers, Nv, Nvp, Ev]
            unified_text_prompt,unified_visual_prompt = self.clip.encode_prompt(batch_size, device=text_ids.device) 
            unified_text_prompt = unified_text_prompt[:, text_labels, :, :] # unified_text_prompt: [Nlayers, Nt, Ntp, Et]
        
        if text_ids is not None:
            sequence_output = self.get_sequence_output(text_ids, unified_text_prompt) 
            output_dict['sequence_output'] = sequence_output

        if video is not None: 
            video = torch.as_tensor(video).float()
            batch_size, video_frame, channel, h, w = video.shape
            visual_output = self.get_visual_output(video,unified_visual_prompt)
            output_dict['visual_output'] = visual_output

        if self.training:
            logits = self.get_prvr_similarity_logits(sequence_output, visual_output)
            losses = self.get_loss(logits, text_labels, v2t_dict,**kwargs)

            output_dict.update(losses)

        return output_dict


    def get_nce_loss(self, sim_matrix, text_labels, v2t_dict, **kwargs):
        Nt, Nv = sim_matrix.shape 
        txt_range_ids = torch.arange(Nt).to(sim_matrix.device)
        t2v_nominator = sim_matrix[txt_range_ids, text_labels] # [Nt]
        t2v_nominator = torch.logsumexp(t2v_nominator.unsqueeze(1), dim=1)
        t2v_denominator = torch.logsumexp(sim_matrix, dim=1)

        v2t_nominator = torch.zeros(Nv).to(sim_matrix.device)
        v2t_denominator = torch.zeros(Nv).to(sim_matrix.device)

        for i, label in v2t_dict.items():
            v2t_nominator[i] = torch.logsumexp(sim_matrix[label, i], dim=0)
            v2t_denominator[i] = torch.logsumexp(sim_matrix[:, i], dim=0)
        

        t2v_loss = torch.mean(t2v_denominator-t2v_nominator)
        v2t_loss = torch.mean(v2t_denominator-v2t_nominator)

        sim_loss = (t2v_loss + v2t_loss) / 2
        return {
            't2v_loss': t2v_loss,
            'v2t_loss': v2t_loss,  
            'sim_loss': sim_loss  
        }

    
    def get_loss(self, logits, text_labels, v2t_dict, **kwargs):
        losses = {} 
        nce_loss_res = self.get_nce_loss(logits, text_labels, v2t_dict,  **kwargs)
        losses.update(nce_loss_res)

        loss = losses['sim_loss'] 
        losses['loss'] = loss 

        return losses 

        
    @torch.no_grad()
    def encode_video(self, video, return_weights=False):
        batch_size = video.size(0)
        # TODO: add separated encode_prompt_video and encode_prompt_text functions
        _, unified_visual_prompt = self.clip.encode_prompt(batch_size,device=video.device)

        video = torch.as_tensor(video).float()
        batch_size, video_frame, channel, h, w = video.shape
        if return_weights:
            visual_output, weights = self.get_visual_output(video,unified_visual_prompt,return_weights=True)
            return visual_output, weights 
        else:
            visual_output = self.get_visual_output(video,unified_visual_prompt) 
            return visual_output
    

    @torch.no_grad()
    def encode_text(self, text_ids):
        batch_size = text_ids.size(0)
        unified_text_prompt, _ = self.clip.encode_prompt(batch_size,device=text_ids.device)
        sequence_output = self.get_sequence_output(text_ids, unified_text_prompt)
        return sequence_output


    def get_sequence_output(self, text_ids,unified_text_prompt):
        bs_pair = text_ids.size(0)
        sequence_hidden = self.clip.encode_text(text_ids,unified_text_prompt).float()
        tFeature = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1)) 
        return tFeature


    def get_visual_output(self,video,unified_visual_prompt, return_weights=False, **kwargs):
        batch_size, video_frame, channel, h, w = video.shape

        video = video.view(-1, channel, h, w) 
        if return_weights:
            visual_hidden, weights = self.clip.encode_image(video,unified_visual_prompt,return_weights=True,**kwargs) 
        else:
            visual_hidden = self.clip.encode_image(video,unified_visual_prompt,**kwargs) 
        vFeature = visual_hidden.view(batch_size, -1, visual_hidden.size(-1))

        if return_weights:
            return vFeature, weights 
        else:
            return vFeature

    
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def get_prvr_similarity_logits(self, sequence_output, visual_output, return_orig_scores=False, **kwargs):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        if self.training or not self.pre_visual_pooling:
            visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True) 

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True) 
        logit_scale = self.clip.logit_scale.exp().to(sequence_output.device)

        _logits_whole = torch.einsum("td,vkd->tvk", sequence_output, visual_output) # [Nt, Nv, Ne]
        _logits, indices = torch.max(_logits_whole, dim=-1) 
        logits = logit_scale * _logits 

        if return_orig_scores:
            return logits, _logits_whole
        else:
            return logits


    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def get_vcmr_similarity_logits(self, sequence_output, visual_output, args=None, **kwargs):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True) 

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True) 

        logits = torch.einsum("td,vkd->tvk", sequence_output, visual_output) 
        npev = args.npev
        sel_scores, sel_indices = torch.topk(logits, npev, dim=2) 
        
        clip_proposals = self.clip.visual.proposals.to(sequence_output.device) 
        clip_proposals = clip_proposals[None, None, :, :].repeat(sel_scores.shape[0], sel_scores.shape[1], 1, 1) 
        sel_indices_ = sel_indices.unsqueeze(-1).repeat(1, 1, 1, 2) 
        sel_props = torch.gather(clip_proposals, dim=2, index=sel_indices_) 
        return sel_scores, sel_props


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(CLIP4Clip_PRVR, self).train(mode)
        no_clip = self.new_added_modules

        if self.freeze_clip and mode:
            logging.info("(model) Freezing ALL the CLIP backbone.")
            # import pdb; pdb.set_trace() 
            for name, param in self.clip.named_parameters():
                if not any(nd in name for nd in no_clip):
                    param.requires_grad = False
                else:
                    pass 
                    # logging.info('trainerble parameters are:{}'.format(name))

            
            for name, m in self.clip.named_modules():
                if not any(nd in name for nd in no_clip):
                    if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.Dropout)):
                        m.eval()
            for m in self.clip.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    m.eval()


    def print_trainable_params_percentage(self):

        orig_param_size = sum(p.numel() for p in self.clip.parameters())

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        trainable_size = count_parameters(self.clip)

        percentage = trainable_size / orig_param_size * 100

        print(f"Trainable param percentage: {percentage:.2f}% ({trainable_size}/{orig_param_size})")

        return percentage
