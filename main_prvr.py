# coding=utf-8
from __future__ import (absolute_import, division, unicode_literals)

import os
import sys
import torch
import logging
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


def main(args):
    """main worker"""
    args.gpu=0
    args.distributed=False # we only support single gpu for now (codes for gathering text labels with dynamic lengths are so cumbersome)
    set_logger(os.path.join(args.output_dir, "log.txt"), args.log_level)
    set_random_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    save_hp_to_json(args.output_dir, args)
    tokenizer = ClipTokenizer()
    model_state_dict = torch.load(args.init_model, map_location='cpu') if args.init_model else None
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    
    model = CLIP4Clip_PRVR.from_pretrained(args.cross_model, 
                                        cache_dir=cache_dir,
                                        state_dict=model_state_dict,
                                        task_config=args)
    if args.precision == "amp"or args.gpu is None:  
        logging.info("[weight convert] ==>> Convert weights to fp32 for {}...".format(args.precision))
        convert_models_to_fp32(model)
        logging.info("[weight convert] ==>> Convert done!")

    if not torch.cuda.is_available():
        model.float()
        logging.warning("using CPU, this will be slow")
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
    train_dataloader, train_length, train_sampler = get_train_dataloader(args, tokenizer)
    
    
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                    / args.gradient_accumulation_steps) * args.epochs

    ## ####################################
    # optimization strategies
    ## ####################################
    optimizer_grouped_parameters = prep_optim_params_groups(args, model, coef_lr=args.coef_lr)
    scaler = GradScaler() if args.precision == "amp" else None
    if args.optim == 'BertAdam':
        logging.info('[optimizer] Using BertAdam Optimizer...')
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                                schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                                t_total=num_train_optimization_steps, weight_decay=args.wd,
                                max_grad_norm=1.0)
        scheduler = None
    elif args.optim == 'AdamW':
        logging.info('[optimizer] Using AdamW Optimizer...')
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr,
                                betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.wd)
        scheduler = lr_scheduler(mode='cos', init_lr=args.lr, all_iters=num_train_optimization_steps,
                                    slow_start_iters=args.warmup_proportion * num_train_optimization_steps,
                                    weight_decay=args.wd
                                )
    else:
        raise NotImplementedError

    ## ####################################
    #  optionally resume from a checkpoint
    ## ####################################
    start_epoch, global_step = 0, 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)

            if not args.load_from_pretrained:
                if "optimizer" in checkpoint and optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "scaler" in checkpoint and scaler is not None:
                    logging.info("[resume] => Loading state_dict of AMP loss scaler")
                    scaler.load_state_dict(checkpoint['scaler'])
                start_epoch, global_step = checkpoint["epoch"], checkpoint["global_step"]

            logging.info(f"[resume] => loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})\n")
        else:
            logging.info("[resume] => no checkpoint found at '{}'\n".format(args.resume))


    ## ####################################
    # train and evalution
    ## ####################################
    logging.info("\n======================== Running training ========================")
    logging.info("  Num examples = %d", train_length)
    logging.info("  Batch size = %d", args.batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)
    logging.info("\n======================== Running val ========================")
    logging.info("  Num videos = %d", val_vis_length)
    logging.info("  Num text = %d", val_txt_length)
    logging.info("  Batch size = %d", args.batch_size_val)
    logging.info("  Num steps = %d", len(val_txt_dataloader))

    if args.do_eval:
        tv_metrics, info_str = prvr_eval_epoch(model, val_vis_dataloader, val_txt_dataloader, device, args=args)
        torch.cuda.synchronize()
        sys.exit(0)

    best_R, best_e = 0, 0 
    best_info = []
    stop_R = 'SumR'
    for epoch in range(start_epoch, args.epochs):
        tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, optimizer,
                                            global_step, scaler=scaler, scheduler=scheduler)

        logging.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

        tv_metrics, info_str = prvr_eval_epoch(model, val_vis_dataloader, val_txt_dataloader, device, args=args)
        
        if best_R <= tv_metrics[stop_R]:
            best_R = tv_metrics[stop_R]
            best_e = epoch
            best_info = info_str
            logging.info('Got New Best Model')
        else:
            logging.info('A relative failed epoch')
        logging.info("The best {} is: {:.4f}, best_e={}\n".format(stop_R, best_R, best_e))
        ckpt_dict = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': 'CLIp4Clip',
                'state_dict': model.state_dict(),
                'best_acc1': best_R,
                'optimizer': optimizer.state_dict(),
            }
        if scaler is not None: ckpt_dict['scaler'] = scaler.state_dict()
        save_checkpoint(ckpt_dict, best_R <= tv_metrics[stop_R], args.output_dir, filename='ckpt.pth.tar')


    logging.info("The best {} is: {:.4f}, best_epoch={}\n".format(stop_R, best_R, best_e))
    for info in best_info:
        logging.info(info)
    print("The above program id is {}\n".format(args.output_dir))

    torch.cuda.empty_cache()
    sys.exit(0)


def train_epoch(epoch, args, model, train_dataloader, device, optimizer, global_step,
                scheduler=None, scaler=None):
    samples_per_epoch = len(train_dataloader.dataset)

    model.train()

    if epoch == 0:
        trainable_size =0
        total_param_size  = 0  
        for name, param in model.named_parameters():
            if param.requires_grad==True:
                total_param_size += param.numel() 
                trainable_size += param.numel() 
                param_size_MB = param.numel()/(1000**2)
                logging.info(f'trainerble parameters are: {name}, size is {param_size_MB:.4f} MB')
            else:
                total_param_size += param.numel() 
        trainable_size_MB = trainable_size/(1000**2)
        total_param_size_MB = total_param_size/(1000**2)
        percentage = (trainable_size / total_param_size)*100
        logging.info("Trainable param percentage are: {}".format(percentage))
        logging.info("Trainable params are: {} MB, Total params are: {} MB".format(trainable_size_MB,total_param_size_MB))

    
    total_loss = 0
    local_rank = get_rank() 

    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=(local_rank!=0)):
        optimizer.zero_grad()
        if scheduler is not None: scheduler(optimizer, global_step=global_step)
        if torch.cuda.is_available():
            for k, v in batch.items():
                if type(v) is torch.Tensor:
                    batch[k] = v.to(device=device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(batch, epoch_id=epoch)
            loss = output['loss'].mean()
            sim_loss = output['sim_loss'].mean()

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()  
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, 0.1, 4.6052)
        else:
            torch.clamp_(model.clip.logit_scale.data, 0.1, 4.6052)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            global_step += 1
            if global_step % args.n_display == 0:
                num_samples = (step + 1) * len(batch['video_name']) * args.world_size
                percent_complete = num_samples * 1.0 / samples_per_epoch * 100
                logit_scale_data = model.module.clip.logit_scale.data if hasattr(model, 'module') \
                                    else model.clip.logit_scale.data
                lr_tmp = optimizer.param_groups[0]['lr'] if args.optim == 'AdamW' else \
                            optimizer.get_lr()[0]
                
                logging.info(
                    f"Epoch: {epoch} [{num_samples} ({percent_complete:.1f}%)]\t"
                    f"SimLoss: {sim_loss.item():.4f} \t"
                    f"\tLR: {lr_tmp:.1e}\tlogit_scale {logit_scale_data:.3f}"
                )
        total_loss += float(loss)

    total_loss = total_loss / len(train_dataloader)

    return total_loss, global_step

def prvr_eval_epoch(model, val_vis_dataloader, val_txt_dataloader,  device, args=None):
    model.eval()
    local_rank = get_rank() 
    GT_TXT2VID = val_txt_dataloader.dataset.txt2vid
    torch.cuda.empty_cache()
    with torch.no_grad():
        text_names, video_names = [], [] 
        batch_sequence_output_list, batch_visual_output_list = [], []

        logging.info("calculating text features...")
        for bid, batch in tqdm(enumerate(val_txt_dataloader), total=len(val_txt_dataloader), disable=(local_rank!=0)):
            for k, v in batch.items():
                if type(v) is torch.Tensor:
                    batch[k] = v.to(device=device, non_blocking=True)
            sequence_output = model.encode_text(batch['text_ids'])
            batch_sequence_output_list.append(sequence_output.cpu()) 
            torch.cuda.empty_cache()
            text_names += batch['text_name'] 

        logging.info("calculating video features...")
        for bid, batch in tqdm(enumerate(val_vis_dataloader), total=len(val_vis_dataloader), disable=(local_rank!=0)):
            batch = tuple(t.to(device=device, non_blocking=True) if type(t) is not tuple else t for t in batch)
            video_name, video = batch
            visual_output = model.encode_video(video)
            batch_visual_output_list.append(visual_output.cpu())
            torch.cuda.empty_cache()
            video_names += video_name
        if torch.cuda.is_available(): torch.cuda.synchronize()
        sim_matrix  = _prvr_run_on_single_gpu(model, batch_sequence_output_list, batch_visual_output_list, args=args)

    logging.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
    tv_metrics = compute_metrics_prvr(sim_matrix, text_names, video_names, gt_txt2vid=GT_TXT2VID)
    logging.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    info_str = []
    info_str.append("Text-to-Video:")
    info_str.append(' (metric) >>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@100: {:.1f} - SumR: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['R100'], tv_metrics['SumR'], tv_metrics['MR'], tv_metrics['MeanR']))
    for info in info_str: logging.info(info)

    return tv_metrics, info_str

def _prvr_run_on_single_gpu(model, batch_sequence_output_list, batch_visual_output_list, args=None):
    if hasattr(model, 'module'):
        model = model.module
    else:
        model = model

    sim_matrix = []

    for sequence_output in batch_sequence_output_list:
        each_row = []
        for visual_output in batch_visual_output_list: 
            b1b2_logits = model.get_prvr_similarity_logits(sequence_output, visual_output)
            torch.cuda.empty_cache()
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    return sim_matrix



if __name__ == "__main__":
    args = get_args()
    main(args)
    