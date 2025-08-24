import os 
import json
import argparse
from tqdm import tqdm 

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def main(cfg):
    orig_data = load_jsonl(cfg.orig_anno_path)
    save_data = []
    for o_data in tqdm(orig_data, total=len(orig_data)):
        qid = o_data["qid"]
        query = o_data["query"]
        duration = o_data["duration"]
        vid = o_data["vid"]
        

        s_data = {
            "vid_name": vid,
            "desc_id": qid, 
            "duration": duration,
            "desc": query
        }

        if "relevant_windows" in o_data: # for train, val
            relevant_window = o_data["relevant_windows"]
            s_data["ts"] = relevant_window

        save_data.append(s_data)
    
    save_jsonl(save_data, cfg.save_anno_path)
    print(f"saved to {cfg.save_anno_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_anno_path", type=str, required=True)
    parser.add_argument("--save_anno_path", type=str, required=True)
    cfg = parser.parse_args()

    main(cfg)