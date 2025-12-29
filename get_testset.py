import json
import os
import random
import shutil
import subprocess
import json
import re

# ... (get_video_duration 函数保持不变) ...

def get_video_duration(video_path):
    """使用 ffprobe 获取视频时长（秒）"""

    FFPROBE_PATH = r"D:\\Internet_Downloads\\ffmpeg-8.0.1-essentials_build\\ffmpeg-8.0.1-essentials_build\\bin\\ffprobe.exe"

    if not os.path.exists(FFPROBE_PATH):
        print(f"❌ ffprobe 未找到：{FFPROBE_PATH}")
        print("➡ 请确认你的 ffmpeg 安装路径是否正确")
        return 0.0

    if not os.path.exists(video_path):
        # print(f"⚠ 文件不存在：{video_path}") 
        return 0.0
    
    def run_ffprobe(cmd):
        try:
            # 使用 shell=True 以避免在 Windows 上可能出现的权限问题
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW # Windows下隐藏命令行窗口
            )
            return result.stdout.strip()
        except:
            return ""

    # ---------- 方法 1: stream duration ----------
    cmd1 = [
        FFPROBE_PATH, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    duration = run_ffprobe(cmd1)
    if duration and duration != "N/A":
        try:
            return float(duration)
        except ValueError:
            pass

    # ---------- 方法 2: format duration ----------
    cmd2 = [
        FFPROBE_PATH, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    duration = run_ffprobe(cmd2)
    if duration and duration != "N/A":
        try:
            return float(duration)
        except ValueError:
            pass

    # ---------- 方法 3: 用帧数与帧率计算 ----------
    cmd3 = [
        FFPROBE_PATH, "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames,r_frame_rate",
        "-of", "default=noprint_wrappers=1",
        video_path
    ]
    output = run_ffprobe(cmd3)

    frames_match = re.search(r"nb_read_frames=(\d+)", output)
    fps_match = re.search(r"r_frame_rate=(\d+)/(\d+)", output)

    if frames_match and fps_match:
        try:
            frames = int(frames_match.group(1))
            fps = float(fps_match.group(1)) / float(fps_match.group(2))
            if fps > 0:
                return frames / fps
        except:
            pass

    # 如果全失败，返回 0
    return 0.0


# -----------------------------
# 配置路径（按你的实际路径修改）
# -----------------------------
# **【修改点 1：使用 validation.json】**
LABEL_JSON = "E:\\DL-final\\something-something-v2-labels.json"
ANNOTATION_JSON = "E:\\DL-final\\something-something-v2-validation.json" # 使用验证集
VIDEO_DIR = "E:\\DL-final\\20bn-something-something-v2"
OUTPUT_DIR = "E:\\DL-final\\test_set" # 更改输出目录名称以反映筛选结果

# -----------------------------
# 三大类动作及其 template 列表 (保持不变)
# -----------------------------
MOVE_ACTIONS = [
    "Moving [something] down",
    "Moving [something] up",
    "Moving [something] away from [something]",
    "Moving [something] closer to [something]",
    "Moving [something] towards the camera",
    "Moving [something] away from the camera",
    "Moving [something] across a surface until it falls down",
    "Moving [something] across a surface without it falling down",
    "Moving [something] and [something] away from each other",
    "Moving [something] and [something] closer to each other",
    "Moving [something] and [something] so they collide with each other",
    "Moving [something] and [something] so they pass each other",
    "Pulling [something] from behind of [something]",
    "Pulling [something] from left to right",
    "Pulling [something] from right to left",
    "Pulling [something] onto [something]",
    "Pulling [something] out of [something]",
    "Pushing [something] from left to right",
    "Pushing [something] from right to left",
    "Pushing [something] off of [something]",
    "Pushing [something] onto [something]",
    "Pushing [something] so that it slightly moves",
    "Taking [something] out of [something]",
]

DROP_ACTIONS = [
    "Dropping [something] onto [something]",
    "Dropping [something] behind [something]",
    "Dropping [something] into [something]",
    "Dropping [something] in front of [something]",
    "Dropping [something] next to [something]",
    "[Something] falling like a feather or paper",
    "[Something] falling like a rock",
    "Lifting [something] up completely, then letting it drop down",
    "Lifting up one end of [something], then letting it drop down",
]

COVER_ACTIONS = [
    "Covering [something] with [something]",
    "Removing [something], revealing [something] behind",
    "Uncovering [something]"
]

CATEGORY_MAP = {
    "move": MOVE_ACTIONS,
    "drop": DROP_ACTIONS,
    "cover": COVER_ACTIONS,
}

# SAMPLES_PER_CLASS = 100 

# -----------------------------
# 加载标注文件 (保持不变)
# -----------------------------
with open(ANNOTATION_JSON, "r", encoding="utf-8") as f:
    annotations = json.load(f)

# -----------------------------
# 为每个大类收集样本 (保持不变)
# -----------------------------
samples_by_category = {cat: [] for cat in CATEGORY_MAP}

for ann in annotations:
    template = ann["template"]
    for category, templ_list in CATEGORY_MAP.items():
        if template in templ_list:
            samples_by_category[category].append(ann)

# -----------------------------
# 为每个大类创建输出目录 + 拷贝并过滤文件
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

for category, samples in samples_by_category.items():
    print(f"\n====== 处理类别：{category} ======")

    # 输出目录，如 subset_filtered_videos_by_category/move
    # **【保留点 3：按类别创建子目录】**
    category_dir = os.path.join(OUTPUT_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    # **【修改点 4：移除抽样和多轮筛选逻辑】**
    # 之前抽样 100 个的复杂逻辑被移除，改为处理所有样本

    chosen = []
    copied = 0
    missing = 0
    filtered_short = 0

    # 打乱样本，让处理顺序更随机
    random.shuffle(samples) 

    print(f"  ▶️ 该类别共有 {len(samples)} 个初始样本，开始过滤和复制...")

    for i, ann in enumerate(samples):
        vid = ann["id"]
        src = os.path.join(VIDEO_DIR, f"{vid}.webm")
        dst = os.path.join(category_dir, f"{vid}.webm")
        
        # 打印进度
        if (i + 1) % 100 == 0:
             print(f"  ▶️ 进度：{i + 1}/{len(samples)} | 已复制 {copied} 个")

        if not os.path.exists(src):
            missing += 1
            continue
        
        # 时长过滤 (duration > 2.0s)
        duration = get_video_duration(src)
        if duration <= 2.0:
            filtered_short += 1
            continue
        
        if copied >= 20:
            continue  # 跳过后续样本
        
        # 复制文件
        try:
            shutil.copy2(src, dst)
            copied += 1
            chosen.append(ann) # 将通过过滤的样本加入最终 JSON 列表
        except Exception as e:
             print(f"⚠ 复制文件 {vid}.webm 失败: {e}")

    print(f"✔ 类别 {category} 处理完毕：")
    print(f"  - 成功复制视频数量: {copied}")
    print(f"  - 原始文件缺失数量: {missing}")
    print(f"  - 时长过滤数量 (<= 2.0s): {filtered_short}")

    # 写入该类自己的 annotations.json
    json_path = os.path.join(category_dir, f"{category}_annotations.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(chosen, jf, indent=2)

    print(f"✔ 已生成 {json_path}")

print("\n🎉 全部分组完成！输出目录：", OUTPUT_DIR)