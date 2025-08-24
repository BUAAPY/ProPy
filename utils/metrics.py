# coding=utf-8
# Evaluation metric for retrieval task
from __future__ import (absolute_import, division, print_function, unicode_literals)

import torch
import numpy as np
import torch.distributed as dist
from .dist_utils import is_dist_avail_and_initialized
from .standalone_eval import eval_retrieval as eval_retrieval_general # used for VCMR

def eval_prvr_q2m(indices, q2m_gts):
    n_q, n_m = indices.shape

    gt_ranks = np.zeros((n_q,), np.int32)
    for i in range(n_q):
        sorted_idxs = indices[i]
        # sorted_idxs = np.argsort(s)
        rank = n_m + 1
        for k in q2m_gts[i]:
            tmp = np.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(np.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(np.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(np.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(np.where(gt_ranks <= 100)[0]) / n_q
    medr = np.median(gt_ranks)
    meanr = gt_ranks.mean()

    return (r1, r5, r10, r100, medr, meanr)

def ap_score(sorted_labels):
    nr_relevant = len([x for x in sorted_labels if x > 0])
    if nr_relevant == 0:
        return 0.0

    length = len(sorted_labels)
    ap = 0.0
    rel = 0

    for i in range(length):
        lab = sorted_labels[i]
        if lab >= 1:
            rel += 1
            ap += float(rel) / (i + 1.0)
    ap /= nr_relevant
    return ap

def prvr_t2v_map(c2i, t2v_gts):
    perf_list = []
    for i in range(c2i.shape[0]):
        d_i = c2i[i, :]
        labels = [0]*len(d_i)

        x = t2v_gts[i][0]
        labels[x] = 1

        sorted_labels = [labels[x] for x in d_i]

        current_score = ap_score(sorted_labels)
        perf_list.append(current_score)
    return np.mean(perf_list)

def get_prvr_gt_idx(video_metas, query_metas, cap2vid):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            # if query_id.split('#', 1)[0] == vid_id:
            if cap2vid[query_id] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt

def compute_metrics_prvr(sim_matrix, text_names, video_names, gt_txt2vid):
    # import pdb; pdb.set_trace() 
    v2t_gt, t2v_gt = get_prvr_gt_idx(video_names, text_names, gt_txt2vid)
    # video retrieval
    t2v_sorted_indices = np.argsort(-sim_matrix, axis=1) # descending
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr) = eval_prvr_q2m(t2v_sorted_indices, t2v_gt)

    t2v_sumr = t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100
    metrics = {
        'R1': t2v_r1,
        'R5': t2v_r5,
        'R10': t2v_r10,
        'R100': t2v_r100,
        'MR': t2v_medr,
        'MeanR': t2v_meanr,
        'SumR': t2v_sumr
    }

    return metrics

def compute_metrics_vcmr(video2idx, predictions, ground_truth, use_desc_type=False, iou_thds=(0.1, 0.3, 0.5, 0.7)):
    """
        video2idx: (dict) {vid_name: vid_id} (from vid dataset)
        predictions: (list(dict)) [
            {
            "desc_id":,int, # from txt dataset
            "desc_name": str, 
            "desc": str, 
            "gt_vid_name":str,
            "gt_ts": [float, float],
            "predictions": [[vid_id(int), st(float), ed(float), score(float),vid_name(str)],[...],...] sorted
            }
        ] 
    
    """
    vcmr_res_dict = {
        'video2idx': video2idx,
        'VCMR': predictions
    }
    metrics = eval_retrieval_general(vcmr_res_dict, ground_truth, iou_thds, use_desc_type=False)
    return metrics 



def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics


def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))


# below two functions directly come from: https://github.com/Deferf/Experiments
def tensor_text_to_video_metrics(sim_tensor, top_k = [1,5,10]):
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)

    # Permute sim_tensor so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sim_matrices = sim_tensor.permute(1, 0, 2)
    first_argsort = torch.argsort(stacked_sim_matrices, dim = -1, descending= True)
    second_argsort = torch.argsort(first_argsort, dim = -1, descending= False)

    # Extracts ranks i.e diagonals
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1 = 1, dim2 = 2))

    # Now we need to extract valid ranks, as some belong to inf padding values
    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(permuted_original_data), torch.isnan(permuted_original_data))
    valid_ranks = ranks[mask]
    # A quick dimension check validates our results, there may be other correctness tests pending
    # Such as dot product localization, but that is for other time.
    #assert int(valid_ranks.shape[0]) ==  sum([len(text_dict[k]) for k in text_dict])
    if not torch.is_tensor(valid_ranks):
        valid_ranks = torch.tensor(valid_ranks)
    results = {f"R{k}": float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks)) for k in top_k}
    results["MedianR"] = float(torch.median(valid_ranks + 1))
    results["MeanR"] = float(np.mean(valid_ranks.numpy() + 1))
    results["Std_Rank"] = float(np.std(valid_ranks.numpy() + 1))
    results['MR'] = results["MedianR"]
    return results


def tensor_video_to_text_sim(sim_tensor):
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)
    # Code to avoid nans
    sim_tensor[sim_tensor != sim_tensor] = float('-inf')
    # Forms a similarity matrix for use with rank at k
    values, _ = torch.max(sim_tensor, dim=1, keepdim=True)

    return torch.squeeze(values).T


def synchronize_meter_between_processes(meter_list=[]):
    """
    synchronize meters between processes
    """
    assert isinstance(meter_list, list)
    for meter in meter_list:
        meter.synchronize_between_processes()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.sum], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]
        self.avg = self.sum / self.count
