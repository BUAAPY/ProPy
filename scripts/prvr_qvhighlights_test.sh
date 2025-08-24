#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
echo "Using local machine for training"

# dataset
dataset=qvhighlights
fps=3

# modify these two dirs
video_dir=/home/panyi/QVHighlightsVideos/compressed_videos
pretrained_dir=CLIP_weights

prvr_train_annos=annotations/qvhighlights_train.jsonl
prvr_val_annos=annotations/qvhighlights_test.jsonl


# train or eval
do_train=1
do_eval=0
resume=None

# for evaluation
# do_train=0
# do_eval=1
# resume=/path/to/ckpt/ckpt.best.pth.tar

# learning strategies
pretrained_clip_name=ViT-B/32
lr=9e-4
coef_lr=1e-3
wd=0.2
epochs=10
optim=AdamW
max_words=64


temperature_new=1.0
load_from_pretrained=0
batch_size=24           
batch_size_val=16
num_workers=8
n_display=40            
precision=amp

freeze_clip=1

shared_latent_space=linear

cfg_name=cfg_32
pyr_config_path="pyr_configs/${cfg_name}.json"

num_adapter_layers=12
temp_kernel_sz=5
adapter_channels=384
Nvp=4

model_dir="logs/prvr_qvhighlights_test"
echo "The model dir is ${model_dir}"

python  main_prvr.py \
        --do_train ${do_train} \
        --do_eval ${do_eval} \
        --num_thread_reader ${num_workers} \
        --epochs ${epochs} \
        --batch_size ${batch_size} \
        --n_display ${n_display} \
        --prvr_train_annos ${prvr_train_annos} \
        --prvr_val_annos ${prvr_val_annos} \
        --video_dir ${video_dir} \
        --output_dir ${model_dir} \
        --optim ${optim} \
        --lr ${lr} \
        --coef_lr ${coef_lr} \
        --wd ${wd} \
        --max_words ${max_words} \
        --batch_size_val ${batch_size_val} \
        --datatype ${dataset} \
        --expand_msrvtt_sentences  \
        --feature_framerate ${fps} \
        --freeze_layer_num 12  \
        --slice_framepos 2 \
        --loose_type \
        --linear_patch 2d \
        --sim_header meanP \
        --pretrained_clip_name ${pretrained_clip_name} \
        --precision ${precision} \
        --pretrained_dir ${pretrained_dir} \
        --freeze_clip ${freeze_clip} \
        --resume ${resume} \
        --load_from_pretrained ${load_from_pretrained} \
        --shared_latent_space ${shared_latent_space} \
        --Nvp ${Nvp} \
        --num_adapter_layers ${num_adapter_layers} \
        --temp_kernel_sz ${temp_kernel_sz} \
        --adapter_channels ${adapter_channels} \
        --pyr_config_path ${pyr_config_path}
echo "Training Finished!!!"

