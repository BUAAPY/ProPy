import os
import cv2
from natsort import natsorted
import threading
from queue import Queue
import argparse

def process_single_video(image_folder, output_path, fps=3):
    images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = natsorted(images)
    
    if not images:
        print(f"Warning: no images found in {image_folder}")
        return
    
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    if first_frame is None:
        print(f"Error: cannot read the first image in {image_folder}")
        return
    
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for img in images:
        frame = cv2.imread(os.path.join(image_folder, img))
        if frame is not None:
            video.write(frame)
    
    video.release()
    print(f"{output_path} DONE")

def batch_convert(root_folder, output_dir, fps=3, max_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    
    sub_folders = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in filenames)
        if has_images and dirpath != root_folder: 
            sub_folders.append(dirpath)
    
    
    work_queue = Queue()
    for folder in sub_folders:
        folder_name = os.path.basename(folder)
        output_path = os.path.join(output_dir, f"{folder_name}.mp4")
        work_queue.put((folder, output_path))
    
    def worker():
        while True:
            try:
                folder, output_path = work_queue.get_nowait()
            except:
                break 
            
            print(f"Processing: {folder}")
            process_single_video(folder, output_path, fps)
            work_queue.task_done()
    
    threads = []
    for _ in range(min(max_workers, len(sub_folders))):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    
    work_queue.join()
    
    for t in threads:
        t.join()
    
    print("Done！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dir", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--max_workers", type=int, default=4)
    cfg = parser.parse_args()                      
    
    batch_convert(cfg.frame_dir, cfg.video_dir, cfg.fps, cfg.max_workers)