# coding=utf-8
from __future__ import (absolute_import, division, unicode_literals)

import sys 
sys.path.append(".")

import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import optim, distributed
from torch.cuda.amp import GradScaler
# from torch.utils.tensorboard import SummaryWriter

from params import get_args, save_hp_to_json
from modules import CLIP4Clip_PRVR, convert_weights
from modules import SimpleTokenizer as ClipTokenizer
from modules.file import PYTORCH_PRETRAINED_BERT_CACHE
from dataloaders.dataset import get_train_dataloader, get_val_txt_dataloader, get_val_vis_dataloader
from utils.lr_scheduler import lr_scheduler
from utils.optimization import BertAdam, prep_optim_params_groups
from utils.log import set_logger
from utils.misc import set_random_seed, convert_models_to_fp32, save_checkpoint
from utils.dist_utils import is_master, get_rank, is_dist_avail_and_initialized, init_distributed_mode
from utils.metrics import compute_metrics_prvr
from tqdm import tqdm 

import shutil
import h5py
import json 


def main(args):
    """main worker"""
    args.gpu=0
    args.distributed=False # we only support single gpu for now (codes for gathering text labels with dynamic lengths are so cumbersome)
    set_random_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tokenizer = ClipTokenizer()
    model_state_dict = torch.load(args.init_model, map_location='cpu') if args.init_model else None
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    
    model = CLIP4Clip_PRVR.from_pretrained(args.cross_model, 
                                        cache_dir=cache_dir,
                                        state_dict=model_state_dict,
                                        task_config=args)
    if args.precision == "amp"or args.gpu is None:  
        convert_models_to_fp32(model)

    if not torch.cuda.is_available():
        model.float()
    else:
        model.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights(model)
        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            if args.freeze_clip:
               print('do freeze_clip:',args.freeze_clip)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                                find_unused_parameters=True if args.freeze_clip else True)
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)

    ## ####################################
    # dataloader loading
    ## ####################################
    val_vis_dataloader, val_vis_length = get_val_vis_dataloader(args)
    val_txt_dataloader, val_txt_length = get_val_txt_dataloader(args, tokenizer)



    ## ####################################
    #  resume from a checkpoint
    ## ####################################

    assert args.resume is not None
    assert os.path.isfile(args.resume)

    if args.gpu is None:
        checkpoint = torch.load(args.resume)
    else:
        loc = "cuda:{}".format(args.gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
    
    sd = checkpoint["state_dict"]
    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)

    prvr_produce_plot_meta(model, val_vis_dataloader, val_txt_dataloader, device, args=args)
    torch.cuda.synchronize()
    sys.exit(0)



def prvr_produce_plot_meta(model, val_vis_dataloader, val_txt_dataloader,  device, args=None):
    os.makedirs(args.plot_dir, exist_ok=True)
    ATTN_PATH = os.path.join(args.plot_dir, "attn.hdf5")
    SCORE_PATH = os.path.join(args.plot_dir, "event_scores.hdf5")
    R1_ID_PATH = os.path.join(args.plot_dir, "r1_ids.json")
    attn_file = h5py.File(ATTN_PATH, "w")
    event_scores_file = h5py.File(SCORE_PATH, "w")


    model.eval()
    local_rank = get_rank() 
    GT_TXT2VID = val_txt_dataloader.dataset.txt2vid
    torch.cuda.empty_cache()
    with torch.no_grad():
        text_names, video_names = [], [] 
        batch_text_names, batch_video_names = [], [] 
        batch_sequence_output_list, batch_visual_output_list = [], []

        print("calculating text features...")
        for bid, batch in tqdm(enumerate(val_txt_dataloader), total=len(val_txt_dataloader), disable=(local_rank!=0)):
            for k, v in batch.items():
                if type(v) is torch.Tensor:
                    batch[k] = v.to(device=device, non_blocking=True)
            sequence_output = model.encode_text(batch['text_ids'])
            batch_sequence_output_list.append(sequence_output.cpu()) 
            torch.cuda.empty_cache()
            text_names += batch['text_name'] 
            batch_text_names.append(batch['text_name'])

        print("calculating video features...")
        all_weights = [] 
        for bid, batch in tqdm(enumerate(val_vis_dataloader), total=len(val_vis_dataloader), disable=(local_rank!=0)):
            batch = tuple(t.to(device=device, non_blocking=True) if type(t) is not tuple else t for t in batch)
            video_name, video = batch
            visual_output, weights = model.encode_video(video, return_weights=True)
            batch_visual_output_list.append(visual_output.cpu())
            weights = weights.cpu().numpy()
            # import pdb; pdb.set_trace()
            all_weights.append(weights)
            torch.cuda.empty_cache()
            video_names += video_name
            batch_video_names.append(video_name)
        
        all_weights = np.concatenate(all_weights, axis=1) # [Nlayers, Nv, Ngv, ...]

        if torch.cuda.is_available(): torch.cuda.synchronize()

        print("producing plot metas...")
        json_ids = []
        sim_matrix, matched_event_scores, matched_ids  = _prvr_run_on_single_gpu(model, batch_sequence_output_list, batch_visual_output_list, batch_video_names, batch_text_names, GT_TXT2VID, args=args)

        for i in range(len(matched_ids)):
            event_scores = matched_event_scores[i] # [Ngv]
            txt_id, vid_name = matched_ids[i]
            local_v_id = video_names.index(vid_name)
            attn_weight = all_weights[:, local_v_id] # [Nlayers, Ngv, ...]

            attn_file[str(txt_id)] = attn_weight
            event_scores_file[str(txt_id)] = event_scores
            json_ids.append([str(txt_id),vid_name]) 

    with open(R1_ID_PATH, "w") as f:
        json.dump(json_ids, f)
    return 

def _prvr_run_on_single_gpu(model, batch_sequence_output_list, batch_visual_output_list, batch_video_names, batch_text_names, GT_TXT2VID, args=None):
    if hasattr(model, 'module'):
        model = model.module
    else:
        model = model
    local_rank = get_rank() 

    sim_matrix = []
    all_video_names, matched_ids, matched_scores = [], [], [] 
    for vid_names in batch_video_names:
        all_video_names += vid_names
    r1_cnt, total_cnt = 0, 0 

    for i, sequence_output in tqdm(enumerate(batch_sequence_output_list), total=len(batch_sequence_output_list), disable=(local_rank!=0) ):
        each_row = []
        all_event_scores = [] 
        for visual_output in batch_visual_output_list: 
            b1b2_logits, event_scores = model.get_prvr_similarity_logits(sequence_output, visual_output, return_orig_scores=True)
            torch.cuda.empty_cache()
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            event_scores = event_scores.cpu().detach().numpy()
            each_row.append(b1b2_logits)
            all_event_scores.append(event_scores)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        event_scores = np.concatenate(tuple(all_event_scores), axis=1)

        tnames = batch_text_names[i]
        assert len(tnames) == len(sequence_output)
        for j, tn in enumerate(tnames):
            vn = GT_TXT2VID[tn]
            vidx = all_video_names.index(vn)
            sidx = np.argmax(each_row[j])
            if sidx == vidx:
                r1_cnt += 1
                matched_ids.append([tn, vn])
                escores = event_scores[j].max(axis=0) 
                matched_scores.append(escores)

        sim_matrix.append(each_row)
        total_cnt += len(tnames)
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    matched_scores = np.stack(matched_scores, axis=0) # [Nx, Ne] (R1 hit samples)

    r1 = r1_cnt / total_cnt
    print("R@1: ", r1)
    
    return sim_matrix, matched_scores, matched_ids



if __name__ == "__main__":
    args = get_args()
    main(args)