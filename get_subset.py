import json
import os
import random
import shutil
import subprocess
import json
import re

# if shutil.which("ffprobe") is None:
#     print("❌ 未检测到 ffprobe，请检查 PATH 配置！")
#     exit(1)
# else:
#     print("✔ ffprobe 检测成功")

FFMPEG_PATH = r"D:\\Internet_Downloads\\ffmpeg-8.0.1-essentials_build\\ffmpeg-8.0.1-essentials_build\bin\\ffmpeg.exe"

def get_video_duration(video_path):
    """使用 ffprobe 获取视频时长（秒）"""

    FFPROBE_PATH = r"D:\\Internet_Downloads\\ffmpeg-8.0.1-essentials_build\\ffmpeg-8.0.1-essentials_build\\bin\\ffprobe.exe"

    if not os.path.exists(FFPROBE_PATH):
        print(f"❌ ffprobe 未找到：{FFPROBE_PATH}")
        print("➡ 请确认你的 ffmpeg 安装路径是否正确")
        return 0.0

    if not os.path.exists(video_path):
        print(f"⚠ 文件不存在：{video_path}")
        return 0.0
    
    def run_ffprobe(cmd):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
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
        return float(duration)

    # ---------- 方法 2: format duration ----------
    cmd2 = [
        FFPROBE_PATH, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    duration = run_ffprobe(cmd2)
    if duration and duration != "N/A":
        return float(duration)

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
        frames = int(frames_match.group(1))
        fps = float(fps_match.group(1)) / float(fps_match.group(2))
        if fps > 0:
            return frames / fps

    # 如果全失败，返回 0
    return 0.0

# print(get_video_duration(r"E:\DL-final\20bn-something-something-v2\46632.webm"))

def resize_video(src, dst):
    """
    使用 ffmpeg 将视频转换为 128x128
    force_original_aspect_ratio 确保比例不变，pad 补黑边
    """
    cmd = [
        FFMPEG_PATH, "-y", "-i", src,
        "-vf", "scale=128:128:force_original_aspect_ratio=increase,crop=128:128", 
        "-an", # 移除音频节省空间
        # "-vcodec", "libx264", # 转换为常用编码提高兼容性
        "-vcodec", "libvpx-vp9",  # 必须改为 VP9 才能存为 webm
        # "-crf", "23", 
        "-crf", "30",             # VP9 的 CRF 值含义与 x264 不同，30 比较适中
        "-b:v", "0",              # 使用 CRF 模式时通常需要配合指定码率为 0
        dst
    ]
    # try:
    #     subprocess.run(cmd, capture_output=True, check=True)
    #     return True
    # except subprocess.CalledProcessError:
    #     return False
    try:
        # 修改点：不再静默处理，捕获错误流
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg 报错详情:\n{e.stderr}") # 这里会告诉你到底是找不到文件、编码不支持还是路径不对
        return False


# -----------------------------
# 配置路径（按你的实际路径修改）
# -----------------------------
LABEL_JSON = "E:\\DL-final\\something-something-v2-labels.json"
ANNOTATION_JSON = "E:\\DL-final\\something-something-v2-validation.json"
VIDEO_DIR = "E:\\DL-final\\20bn-something-something-v2"
OUTPUT_DIR = "E:\\DL-final\\subset_test_128_videos"

# -----------------------------
# 三大类动作及其 template 列表
# -----------------------------
MOVE_ACTIONS = [
    "Moving [something] down",
    "Moving [something] up",
    "Moving [something] away from [something]",
]

DROP_ACTIONS = [
    "Dropping [something] onto [something]",
    "Dropping [something] behind [something]",
    "Dropping [something] into [something]",
]

COVER_ACTIONS = [
    "Covering [something] with [something]",
]

CATEGORY_MAP = {
    "move": MOVE_ACTIONS,
    "drop": DROP_ACTIONS,
    "cover": COVER_ACTIONS,
}

SAMPLES_PER_CLASS = 20



# -----------------------------
# 加载标注文件
# -----------------------------
with open(ANNOTATION_JSON, "r", encoding="utf-8") as f:
    annotations = json.load(f)

# -----------------------------
# 为每个大类收集样本
# -----------------------------
samples_by_category = {cat: [] for cat in CATEGORY_MAP}

for ann in annotations:
    template = ann["template"]
    for category, templ_list in CATEGORY_MAP.items():
        if template in templ_list:
            samples_by_category[category].append(ann)

# -----------------------------
# 为每个大类创建输出目录 + 抽样 + 保存 JSON
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

for category, samples in samples_by_category.items():
    print(f"\n====== 处理类别：{category} ======")

    # 输出目录，如 subset_300_videos/move
    category_dir = os.path.join(OUTPUT_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    # 按需求抽取 100 个
    random.shuffle(samples)  # 先洗牌方便多轮筛选

    chosen = []
    used = set()

    pass_id = 1
    while len(chosen) < SAMPLES_PER_CLASS and len(used) < len(samples):
        print(f"  ▶️ 第 {pass_id} 轮筛选... 已找到 {len(chosen)}/{SAMPLES_PER_CLASS}")
        for ann in samples:
            vid = ann["id"]
            if vid in used:
                continue
            used.add(vid)

            src = os.path.join(VIDEO_DIR, f"{vid}.webm")
            if not os.path.exists(src):
                continue
            
            duration = get_video_duration(src)
            if duration > 2.0:
                chosen.append(ann)

            if len(chosen) >= SAMPLES_PER_CLASS:
                break
        pass_id += 1

    print(f"✔ 最终筛选到 {len(chosen)} 条有效样本")

    # 拷贝文件 & 记录 JSON
    copied = 0
    # missing = 0
    out_json = []

    for ann in chosen:

        vid = ann["id"]
        src = os.path.join(VIDEO_DIR, f"{vid}.webm")
        dst = os.path.join(category_dir, f"{vid}.webm")

        if os.path.exists(src):
            # shutil.copy2(src, dst)
            # copied += 1
            # out_json.append(ann)
            # 调用 ffmpeg 进行处理
            if resize_video(src, dst): 
                copied += 1
                out_json.append(ann)
            else:
                print(f"处理失败: {vid}")

        print(f" 类别 {category}: 成功复制 {copied}/{SAMPLES_PER_CLASS} 个视频")

        # duration = get_video_duration(src)
        # if duration <= 2.0:
        #     print(f"跳过 {vid}：视频时长仅 {duration:.2f}s")
        #     continue

        # shutil.copy2(src, dst)
        # copied += 1
        # out_json.append(ann)

        # if os.path.exists(src):
        #     shutil.copy2(src, dst)
        #     copied += 1
        #     # out_json.append({
        #     #     "id": vid,
        #     #     "template": ann["template"]
        #     # })
        #     out_json.append(ann)
        # else:
        #     missing += 1

    # print(f"✔ 成功复制 {copied} 个视频, 缺失 {missing} 个文件。")

    # 写入该类自己的 annotations.json
    json_path = os.path.join(category_dir, f"{category}_annotations.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(out_json, jf, indent=2)

    print(f"✔ 已生成 {json_path}")

print("\n🎉 全部分组完成！输出目录：", OUTPUT_DIR)
