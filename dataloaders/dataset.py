# coding=utf-8
"""dataset for MSRVTT
"""
from __future__ import absolute_import, division, unicode_literals

import os
import json
import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from .decode import RawVideoExtractorpyAV
from .utils import load_jsonl
from torch.utils.data import DataLoader


class Val_VideoDataset(Dataset):
    def __init__(
            self,
            jsonl_path,
            video_dir,
            max_frames=100,
            image_resolution=224,
    ):
        self.data = load_jsonl(jsonl_path)
        self.video_dir = video_dir
        self.max_frames = max_frames
        self.video_names = [] # as a set
        self.vidname2vidid = {}
        self.vidname2duration = {} 
        
        vid_cnt = 0 
        for itm in self.data:
            vid_name = itm['vid_name']
            if vid_name not in self.video_names:
                self.video_names.append(vid_name)
                self.vidname2vidid[vid_name] = vid_cnt
                vid_cnt += 1
                self.vidname2duration[vid_name] = itm['duration']

        self.rawVideoExtractor = RawVideoExtractorpyAV(size=image_resolution, is_train=False,
                                                        num_segments=self.max_frames)
    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video, slice_len = self.rawVideoExtractor.get_video_data(os.path.join(self.video_dir, "{}.mp4".format(video_name))) 
        return video_name, video 



class Val_TextDataset(Dataset):
    def __init__(
            self,
            jsonl_path,
            tokenizer,
            max_words=30,
    ):
        self.data = load_jsonl(jsonl_path) 
        self.txt2vid = {} 
        self.descname2desc = {}
        self.descname2descid = {} 
        self.descname2ts = {} 
        for desc_id, itm in enumerate(self.data): 
            self.txt2vid[itm['desc_id']] = itm['vid_name']
            self.descname2desc[itm['desc_id']] = itm['desc']
            self.descname2descid[itm['desc_id']] = desc_id
            if 'ts' in itm: # qv_test does not have ts
                self.descname2ts[itm['desc_id']] = itm['ts']
        self.max_words = max_words
        self.tokenizer = tokenizer

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.data)



    def __getitem__(self, idx):
        sentence = self.data[idx]['desc']
        text_name = self.data[idx]['desc_id']
        words = self.tokenizer.tokenize(sentence) 

        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words 
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]] 

        input_ids = self.tokenizer.convert_tokens_to_ids(words) 
        text_ids = torch.tensor(input_ids, dtype=int)
        return text_name, text_ids,


class TrainDataset(Dataset):
    def __init__(
            self,
            jsonl_path,
            video_dir,
            tokenizer,
            max_words=30,
            max_frames=100,
            image_resolution=224,

    ):
        self.data = load_jsonl(jsonl_path) 
        self.video_dir = video_dir
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        

        self.sample_len = 0

        self.video_names = [] 
        self.vid2text = {}
        self.text_dict = {}

        for itm in self.data:
            vid_name, txt_name, txt = itm['vid_name'], itm['desc_id'], itm['desc']
            if vid_name not in self.video_names:
                self.video_names.append(vid_name)
            
            if vid_name not in self.vid2text:
                self.vid2text[vid_name] = [txt_name]
            else:
                self.vid2text[vid_name].append(txt_name)
            
            self.text_dict[txt_name] = txt 
        
        self.sample_len = len(self.video_names)

        self.rawVideoExtractor = RawVideoExtractorpyAV(size=image_resolution,
                                                        num_segments=self.max_frames)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len


    def _get_text(self, captions):

        texts_ids = []
        for caption in captions:
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words 
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]] 

            txt_ids = self.tokenizer.convert_tokens_to_ids(words) 
            
            texts_ids.append(torch.tensor(txt_ids, dtype=int))


        return texts_ids

    def _get_rawvideo(self, video_name):   
        video, slice_len = self.rawVideoExtractor.get_video_data(os.path.join(self.video_dir, "{}.mp4".format(video_name))) 
        return video 

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video = self._get_rawvideo(video_name)
        texts_name = self.vid2text[video_name]
        texts = [self.text_dict[n] for n in texts_name]
        texts_ids = self._get_text(texts)
        return video_name, video, texts_name, texts_ids


def collate_train(data):
    video_name, video, texts_name, texts_ids = zip(*data)
    video = torch.stack(video, dim=0) 
    merge_texts, all_lengths, text_labels, all_texts_name = [], [], [], []

    for idx, txts in enumerate(texts_ids):
        merge_texts.extend(t for t in txts)
        all_lengths.extend(len(t) for t in txts)
        text_labels.extend(idx for _ in range(len(txts)))
        all_texts_name.extend(n for n in texts_name[idx])
    
    all_texts_ids = torch.zeros((len(all_lengths), max(all_lengths)), dtype=int)

    for idx, txt in enumerate(merge_texts):
        end = all_lengths[idx]
        all_texts_ids[idx, :end] = txt 
    
    text_labels = torch.tensor(text_labels, dtype=int)
    return dict(
        video_name = video_name,
        video = video, 
        text_name = all_texts_name,
        text_ids = all_texts_ids,
        text_labels = text_labels
    )

def collate_val_text(data):
    text_name, text_ids = zip(*data)
    all_lengths = [t.shape[0] for t in text_ids]
    padded_texts = torch.zeros((len(all_lengths), max(all_lengths)), dtype=int)
    for i, e in enumerate(all_lengths):
        padded_texts[i][:e] = text_ids[i]
    
    return dict(
        text_name=text_name,
        text_ids=padded_texts
    )


def get_train_dataloader(args, tokenizer):
    train_dataset = TrainDataset(
        jsonl_path=args.prvr_train_annos,
        video_dir=args.video_dir,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.Nf
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_train
    )

    return dataloader, len(train_dataset), train_sampler


def get_val_vis_dataloader(args):
    val_vis_dataset = Val_VideoDataset(
        jsonl_path=args.prvr_val_annos,
        video_dir=args.video_dir,
        max_frames=args.Nf
    )

    dataloader = DataLoader(
        val_vis_dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )

    return dataloader, len(val_vis_dataset)


def get_val_txt_dataloader(args, tokenizer):
    val_txt_dataset = Val_TextDataset(
        jsonl_path=args.prvr_val_annos,
        max_words=args.max_words,
        tokenizer=tokenizer,
    )


    dataloader = DataLoader(
        val_txt_dataset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_val_text
    )

    return dataloader, len(val_txt_dataset)
