# Charades
raw_video_path=/home/panyi/Charades_v1
compressed_video_path=/home/panyi/Charades_dataset/compressed_videos
python preprocess/compress_video.py --input_root $raw_video_path --output_root $compressed_video_path --dataset cha

# ActivityNet: v1-2/train, v1-2/val, v1-3/train_val
raw_video_path=/home/panyi/ActivityNetDataset/v1-2/train
compressed_video_path=/home/panyi/ActivityNetDataset/compressed_videos
python preprocess/compress_video.py --input_root $raw_video_path --output_root $compressed_video_path --dataset act

raw_video_path=/home/panyi/ActivityNetDataset/v1-2/val
compressed_video_path=/home/panyi/ActivityNetDataset/compressed_videos
python preprocess/compress_video.py --input_root $raw_video_path --output_root $compressed_video_path --dataset act


raw_video_path=/home/panyi/ActivityNetDataset/v1-3/train_val
compressed_video_path=/home/panyi/ActivityNetDataset/compressed_videos
python preprocess/compress_video.py --input_root $raw_video_path --output_root $compressed_video_path --dataset act

# # TVR: put all frame directions under raw_frame_path; convert to videos first; then compress
raw_frame_path=/home/panyi/TVRDataset/frames
raw_video_path=/home/panyi/TVRDataset/videos
python preprocess/image_to_mp4.py --frame_dir $raw_frame_path --video_dir $raw_video_path
compressed_video_path=/home/panyi/TVRDataset/compressed_videos
python preprocess/compress_video.py --input_root $raw_video_path --output_root $compressed_video_path --dataset tvr


# # QVHighlights
raw_video_path=/home/panyi/QVHighlightsVideos/videos
compressed_video_path=/home/panyi/QVHighlightsVideos/compressed_videos
python preprocess/compress_video.py --input_root $raw_video_path --output_root $compressed_video_path --dataset qv

# converted annotations are provided
# train_anno_path=/home/panyi/moment_detr/data/highlight_train_release.jsonl
# save_train_anno_path=/home/panyi/QVHighlightsVideos/TextData/highlight_train_release.jsonl
# python preprocess/convert_qv_annos.py --orig_anno_path $train_anno_path --save_anno_path $save_train_anno_path

# val_anno_path=/home/panyi/moment_detr/data/highlight_val_release.jsonl
# save_val_anno_path=/home/panyi/QVHighlightsVideos/TextData/highlight_val_release.jsonl
# python preprocess/convert_qv_annos.py --orig_anno_path $val_anno_path --save_anno_path $save_val_anno_path

# test_anno_path=/home/panyi/moment_detr/data/highlight_test_release.jsonl
# save_test_anno_path=/home/panyi/QVHighlightsVideos/TextData/highlight_test_release.jsonl
# python preprocess/convert_qv_annos.py --orig_anno_path $test_anno_path --save_anno_path $save_test_anno_path
