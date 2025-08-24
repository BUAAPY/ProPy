import sys 
sys.path.append(".")
import json 
import math 
import argparse
from dataloaders.decode import RawVideoExtractorpyAV
import os 
from torchvision.transforms import ToPILImage
import h5py 
from tqdm import tqdm 
import torch
import numpy as np 
from PIL import Image
import cv2
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import seaborn as sns 


palette = sns.color_palette("Blues", 10) 

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--plot_dir", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--pyr_config_path", type=str, default="pyr_configs/cfg_32.json")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--Nvp", type=int, default=4, help="num of visual prompts")
    parser.add_argument("--grid_size", type=int, default=7, help="CLIP image grid size")
    cfg = parser.parse_args()
    return cfg 

def load_annos(cfg):
    metas = {} 
    with open(cfg.anno_path, "r") as f:
        for line in f.readlines():
            d = json.loads(line)
            qid = d["desc_id"]
            duration = d["duration"]
            metas[str(qid)] = {"desc": d['desc'] , "vid_name": d['vid_name'],}
            if "ts" in d: # ts not available for qv_test
                ts = d["ts"]
                if type(ts[0]) is list: # multiple. select the first one
                    ts = ts[0]
                st = int(ts[0]/duration * cfg.num_frames)
                ed = int(ts[1]/duration * cfg.num_frames)
                metas[str(qid)]["gt"] = [st, ed]
            
    return metas 


def get_pyr_configs(cfg):
    with open(cfg.pyr_config_path, "r") as f:
        pyr_config = json.load(f)
    Nf = pyr_config["Nf"]
    cfg.num_frames = Nf
    if "H" in pyr_config:
        H = pyr_config["H"]
    elif "offset" in pyr_config:
        # another improved and easier construction method. 
        # 1 for dense (offset=seg_len//2), 0 for sampled (offset=seg_len) for each layer
        # will be detailed in our journal version
        offset = pyr_config["offset"] 
        sel_layers = [(0, 1, Nf)] 
        p = int(math.log(Nf, 2))
        assert 2**p == Nf
        for i in range(1, p+1):
            seg_len = 2 ** i
            k = seg_len-1 
            if k == Nf-1:
                sel_layers.append((k, 1, 1))
            else:
                if offset[i-1] == 1:
                    ok0 = seg_len // 2
                elif offset[i-1] == 0:
                    ok0 = seg_len
                else:
                    raise NotImplementedError
                
                assert (Nf-seg_len) % ok0 == 0
                nk = (Nf-seg_len)//ok0+1
                sel_layers.append((k, ok0, nk))

        structure = []
        m = len(sel_layers)
        for i in range(m-1):
            k2, ok2, nk2 = sel_layers[i]
            k1, ok1, nk1 = sel_layers[i+1] # k1>k2
            c_k1k2 = (k1-k2)//ok2 + 1
            if nk1 == 1:
                o_k1k2 = 1
            else:
                o_k1k2 = (nk2-c_k1k2)//(nk1-1)
            structure.append([c_k1k2, o_k1k2])
        H = structure
    else:
        raise NotImplementedError
    
    num_prompt_layers = len(H)
    kernel_and_offset = {}
    cnts = [Nf] # raw frames (0-th layer)
    for k in range(num_prompt_layers, 0, -1):
        kernel_and_offset[(k, k-1)] = H[k-1]
    
        
    for k in range(num_prompt_layers):
        n_prev = cnts[-1]
        c, o = H[k]
        assert (n_prev - c) % o == 0
        n = (n_prev - c) // o + 1
        cnts.append(n)
    
    level_event_cnts = cnts[1:][::-1] # top first
    
    for k1 in range(num_prompt_layers, 0, -1): 
        for k2 in range(k1-1, -1, -1):
            if k2==k1-1:
                continue
            else:
                o_prev = kernel_and_offset[(k1, k2+1)][1]
                o_k1k2 = o_prev * kernel_and_offset[(k2+1, k2)][1]
                c_k1k2 = cnts[k2] - o_k1k2 * (cnts[k1] - 1)
                kernel_and_offset[(k1, k2)] = [c_k1k2, o_k1k2]
    proposals = []
    for l_i, n_i in enumerate(level_event_cnts):
        k, s = kernel_and_offset[(num_prompt_layers-l_i, 0)]
        for i in range(n_i):
            st_i = (i * s)
            ed_i = (i * s + k) 
            proposals.append([st_i, ed_i])

    cfg = {
        "Ne": sum(level_event_cnts), # 42,
        "n1": level_event_cnts[-1], # 16
        "Nf": Nf, # 32
        "proposals": proposals,
        "level_event_cnts": level_event_cnts,
        "H": H
    }
    return cfg


def split_attn(attn_scores, Ne, Nvp, n1, Nf, grid_size):
    # attn_scores: [Nlayers, Ne, Ne+n1*Nvp+(G**2+1)*Nf]
    Nlayers = attn_scores.shape[0]
    attn_ee = attn_scores[:, :, :Ne] # [Nlayers, Ne, Ne]
    clip_v_len = grid_size**2 + 1
    attn_ep = attn_scores[:, :, Ne+Nvp*n1:].reshape(Nlayers, Ne, clip_v_len, Nf) # [Nlayers, Ne, 50, Nf]
    attn_ep = attn_ep[:, :, 1:, :].reshape(Nlayers, Ne, grid_size, grid_size, Nf).permute(0, 1, 4, 2, 3) # [Nlayers, Ne, Nf, G, G]
    return attn_ee, attn_ep


def draw_tree(score_path, save_path, event_id, level_event_cnts, Cs, Os):
    scores = np.load(score_path) # [Ngv]
    orig_ids = np.arange(len(scores))
    scores = (scores - scores.min()) / (scores.max()-scores.min())
    
    orig_ids = orig_ids[::-1]

    st, ed = [0], []
    for k in level_event_cnts:
        ed.append(st[-1]+k)
        st.append(ed[-1])
    
    # suit for picture
    scores = scores[::-1]  # [41, 40, 39, ..., 0]
    for s, e in zip(st, ed):
        scores[s:e] = scores[s:e][::-1] # [26, 27, ..., 41, ...] (reverse in each layer)
        orig_ids[s:e] = orig_ids[s:e][::-1]

    K = len(level_event_cnts)
    # some plot parameters
    RH=1
    RW=1
    DELTA=1.0
    WIDTH=level_event_cnts[0] * RW 
    HEIGHT=K * RH + (K-1) * DELTA + 0.8


    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    previous_left_xs = []

    idx = 0 
    for k in range(K):
        prevxs = [] 
        y = k * (RH + DELTA)
        if k == 0:
            for j in range(level_event_cnts[k]):
                value = scores[idx]
                index = int(value * (len(palette) - 1))  
                color_rgb = palette[index]  
                if orig_ids[idx] == event_id:
                    edge_color = 'red'
                    linewidth = 8
                    zorder = 10
                elif value == scores.max():
                    edge_color = 'orange'
                    linewidth = 8
                    zorder = 10
                else:
                    edge_color = 'black'
                    linewidth = 2
                    zorder = 2

                x = j * RW 
                rect = patches.Rectangle((x, y), RH, RW, linewidth=linewidth, edgecolor=edge_color, facecolor=color_rgb, zorder=zorder)
                ax.add_patch(rect)
                prevxs.append(x)
                idx += 1
            previous_left_xs.append(prevxs)
        else:
            kernel = Cs[k]
            stride = Os[k]
            for j in range(level_event_cnts[k]):
                value = scores[idx]
                index = int(value * (len(palette) - 1)) 
                color_rgb = palette[index] 
                if orig_ids[idx] == event_id:
                    edge_color = 'red'
                    linewidth = 8
                    zorder = 10
                elif value == scores.max():
                    edge_color = 'orange'
                    linewidth = 8
                    zorder = 10
                else:
                    edge_color = 'black'
                    linewidth = 2
                    zorder = 2
                _px = previous_left_xs[-1][j*stride:j*stride+kernel]
                x = sum(_px) / kernel 
                rect = patches.Rectangle((x, y), RH, RW, linewidth=linewidth, edgecolor=edge_color, facecolor=color_rgb, zorder=zorder)
                ax.add_patch(rect)
                prevxs.append(x)
                idx += 1
                xr = x + RW 
                _px_l = _px[0]
                _px_r = _px[-1] + RW 
                _py = (k-1) * (RH + DELTA) + RH 
                ax.plot([_px_l, x], [_py, y], 'k-', linewidth=1)
                ax.plot([_px_r, xr], [_py, y], 'k-', linewidth=1)
            previous_left_xs.append(prevxs)     

    ax.set_xlim(0, 16)
    ax.set_ylim(0, int(HEIGHT))
    ax.set_xticks([])  
    ax.set_yticks([]) 
    ax.set_axis_off()  
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close() 

    
def plot_attn(cfg):
    pyr_cfg = get_pyr_configs(cfg)
    video_extractor = RawVideoExtractorpyAV(size=cfg.resolution, num_segments=pyr_cfg["Nf"], is_train=False)

    tvids = json.load(open(os.path.join(cfg.plot_dir, "r1_ids.json"), "r"))
    metas = load_annos(cfg)
    attn_weights = h5py.File(os.path.join(cfg.plot_dir, "attn.hdf5"), "r")
    event_scores = h5py.File(os.path.join(cfg.plot_dir, "event_scores.hdf5"), "r")
    root_dir = os.path.join(cfg.plot_dir, "attention") 

    to_pil = ToPILImage() 
    eventid_to_proposal = pyr_cfg['proposals']
    Ne, n1, Nf = pyr_cfg["Ne"], pyr_cfg["n1"], pyr_cfg["Nf"]
    level_event_cnts = pyr_cfg["level_event_cnts"][::-1] # from bottom to top
    Cs, Os = zip(*pyr_cfg["H"])

    print("plotting...")

    for qid, vid in tqdm(tvids, total=len(tvids)):
        vid_path = os.path.join(cfg.video_dir, f"{vid}.mp4")
        frames = video_extractor.get_video_data(vid_path)[0]
        meta = metas[str(qid)]
        t_dir = os.path.join(root_dir, qid)
        fig_dir = os.path.join(t_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)


        attn_w = attn_weights[str(qid)][...] # [Nlayers, Ne, ...]
        attn_w = torch.from_numpy(attn_w)
        attn_ee, attn_ep = split_attn(attn_w, Ne=Ne, Nvp=cfg.Nvp, n1=n1, Nf=Nf, grid_size=cfg.grid_size) 
        attn_ee = np.array(attn_ee[-1]) # [Ne, Ne]
        attn_ep = np.array(attn_ep[-1]) # [Ne, Nf, G, G]

        event_s = event_scores[str(qid)][...] # [Ne]
        sel_idx = event_s.argmax()
        meta['event_id'] = sel_idx.item()
        sel_prop = eventid_to_proposal[sel_idx]
        meta['pred_prop'] = sel_prop
        sel_ee = attn_ee[sel_idx] # [Ne]
        score_path = os.path.join(t_dir, "event.npy")
        np.save(score_path, sel_ee)
        sel_ep = attn_ep[sel_idx] # [Nf, G, G]
        sel_ep = (sel_ep - sel_ep.min()) / (sel_ep.max()-sel_ep.min()+1e-6) # normed

        with open(os.path.join(t_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

        for i in range(Nf):
            frame = frames[i]
            frame = (frame - frame.min()) / (frame.max() - frame.min()+1e-6)
            # import pdb; pdb.set_trace() 
            image = to_pil(frame) # make it dark

            attn = sel_ep[i] # [G, G]
            attn = Image.fromarray(attn).resize((cfg.resolution,cfg.resolution), Image.BILINEAR)
            attn = np.array(attn)
            heatmap = cv2.applyColorMap((attn * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = Image.fromarray(heatmap)
            image = Image.blend(image, heatmap, alpha=0.5)
            image.save(os.path.join(fig_dir, f"{i}.jpg"))
        
        # tree
        tree_path = os.path.join(t_dir, "tree.jpg")
        draw_tree(score_path, tree_path, meta['event_id'], level_event_cnts, Cs, Os)
    
    print(f"{len(tvids)} samples selected, figures saved to {root_dir}")

if __name__ == "__main__":
    cfg = get_config()
    plot_attn(cfg)
