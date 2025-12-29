import os
import json
import numpy as np
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import h5py

# ================== CONFIG ===================
DATASET_ROOT = "subset_300_videos"  # 数据集根目录
SAVE_AS = "something_subset.npz"     # 或 "something_subset.h5"
TARGET_SIZE = 96                     # 调整图像大小（TARGET_SIZE x TARGET_SIZE）
SAVE_H5 = False                      # True 保存为 h5，False 保存为 npz
# ==============================================


def extract_frames(video_path, fps_target_frame=21):
    """
    提取视频的第1帧和指定帧（默认21帧）
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Cannot open video {video_path}")
            return None, None

        frames = []
        frame_idx = 0

        first_frame = None
        target_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if frame_idx == 1:
                first_frame = frame

            if frame_idx == fps_target_frame:
                target_frame = frame
                break  # 不再继续读

        cap.release()
        return first_frame, target_frame

    except Exception as e:
        print(f"[Error] Failed reading: {video_path}, {e}")
        return None, None


def load_json(json_path):
    """
    读取动作描述 JSON
    """
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    return data.get("template", "")


def process_split(split_dir):
    """
    处理单个 split（如 cover / drop / move）
    要求 split_dir 中有一个 *_annotations.json 文件
    """

    # ---------- 1. 找到 annotations.json ----------
    annotation_file = [f for f in os.listdir(split_dir) if f.endswith("_annotations.json")]
    if len(annotation_file) == 0:
        print(f"[Error] No annotation json in {split_dir}")
        return [], [], []

    annotation_path = os.path.join(split_dir, annotation_file[0])

    # 加载 annotations
    with open(annotation_path, "r") as f:
        annotations = json.load(f)

    # 根据视频 id 建立索引
    ann_map = {item["id"]: item for item in annotations}

    # ---------- 2. 遍历视频文件 ----------
    all_first = []
    all_future = []
    all_text = []

    for fname in os.listdir(split_dir):
        if not fname.endswith(".webm"):
            continue

        video_id = os.path.splitext(fname)[0]
        video_path = os.path.join(split_dir, fname)

        # ---------- 3. 找到对应 annotation ----------
        if video_id not in ann_map:
            print(f"[Warning] Missing annotation for video {video_id}")
            continue

        text = ann_map[video_id]["label"]

        # ---------- 4. 提取帧 ----------
        first_frame, future_frame = extract_frames(video_path)
        if first_frame is None or future_frame is None:
            print(f"[Warning] Frame extraction failed for {video_id}")
            continue

        # resize + BGR→RGB
        first_frame = cv2.resize(first_frame, (TARGET_SIZE, TARGET_SIZE))
        future_frame = cv2.resize(future_frame, (TARGET_SIZE, TARGET_SIZE))
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        future_frame = cv2.cvtColor(future_frame, cv2.COLOR_BGR2RGB)

        # 保存结果
        all_first.append(first_frame)
        all_future.append(future_frame)
        all_text.append(text)

    return all_first, all_future, all_text


def main():
    splits = ["cover", "drop", "move"]

    FirstFrames = []
    FutureFrames = []
    TextDescriptions = []

    for s in splits:
        split_dir = os.path.join(DATASET_ROOT, s)
        if not os.path.exists(split_dir):
            print(f"[Warning] Folder missing: {split_dir}")
            continue

        print(f"Processing: {split_dir}...")
        f1, f21, text = process_split(split_dir)

        FirstFrames.extend(f1)
        FutureFrames.extend(f21)
        TextDescriptions.extend(text)

    FirstFrames = np.array(FirstFrames)
    FutureFrames = np.array(FutureFrames)

    print("Total samples:", len(TextDescriptions))

    # ---------- Save ----------
    if SAVE_H5:
        with h5py.File(SAVE_AS, "w") as f:
            f.create_dataset("first_frame", data=FirstFrames)
            f.create_dataset("future_frame", data=FutureFrames)
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("text", data=np.array(TextDescriptions, dtype=object), dtype=dt)
        print(f"Saved to {SAVE_AS} (h5 format)")
    else:
        np.savez_compressed(SAVE_AS,
                            first_frame=FirstFrames,
                            future_frame=FutureFrames,
                            text=np.array(TextDescriptions, dtype=object))
        print(f"Saved to {SAVE_AS} (npz format)")


if __name__ == "__main__":
    main()
