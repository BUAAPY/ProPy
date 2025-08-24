#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
echo "Using local machine for training"

# dataset
dataset=activity
fps=3

# modify these two dirs
video_dir=/home/panyi/ActivityNetDataset/compressed_videos # video dirs
pretrained_dir=CLIP_weights # path to store CLIP weights
plot_dir=VIS/activitynet

prvr_train_annos=annotations/activitynet_train.jsonl
prvr_val_annos=annotations/activitynet_val.jsonl


# train or eval
# do_train=1
# do_eval=0
# resume=None

# for evaluation
do_train=0
do_eval=1
resume=logs/prvr_activitynet/ckpt.best.pth.tar



# learning strategies
pretrained_clip_name=ViT-B/32
lr=1e-3
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


pyr_config_path=pyr_configs/cfg_32.json

num_adapter_layers=12
temp_kernel_sz=5
adapter_channels=384
Nvp=4

model_dir="logs/prvr_activitynet"
echo "The model dir is ${model_dir}"


python  visualization/produce_plot_meta.py \
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
        --pyr_config_path ${pyr_config_path} \
        --plot_dir ${plot_dir}


resolution=224
grid_size=7
python visualization/plot_attn.py \
    --video_dir ${video_dir} \
    --plot_dir ${plot_dir} \
    --anno_path ${prvr_val_annos} \
    --pyr_config_path ${pyr_config_path} \
    --resolution ${resolution} \
    --Nvp ${Nvp} \
    --grid_size ${grid_size}
echo "Visualization Finished!!!"

