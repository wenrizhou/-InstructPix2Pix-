# Video Frame Prediction for Human-Object Interaction (Something-Something V2)

This project studies **future frame prediction** on the **Something-Something V2** dataset: generating the **21st frame** conditioned on the first **20 frames** and an action text label. We build a diffusion-based pipeline using **InstructPix2Pix** as the backbone, integrate **ControlNet** for structural control, and fine-tune with **LoRA** for efficiency. We further propose an **improved Temporal Attention** module to better capture multi-frame dynamics.

本项目研究 **Something-Something V2** 数据集上的**下一帧预测**任务：给定前 **20 帧**视频与动作文本标签，生成**第 21 帧**。我们采用扩散模型框架，以 **InstructPix2Pix** 为主干，结合 **ControlNet** 做结构约束，并使用 **LoRA** 进行参数高效微调。同时提出并验证了改进版 **Temporal Attention** 模块，用于更好地建模多帧时序信息。

---

## Task & Motivation

**Goal:** predict a plausible next frame with consistent geometry and semantics under human-object interactions. Pure single-frame conditioning often fails when motion/interaction cues are ambiguous, so multi-frame temporal modeling is critical.

**目标：**在复杂的人-物交互场景下生成几何与语义一致的下一帧。仅依赖单帧（例如第20帧）容易缺乏运动线索，因此需要显式建模多帧时序信息来提升预测的稳定性与合理性。

---

## Method Overview

We adopt an **InstructPix2Pix + ControlNet** pipeline. The model uses (1) visual input frames, (2) a structural control signal (Canny edges), and (3) an action text prompt to synthesize the target frame. To reduce training cost, we apply **LoRA** to the U-Net and ControlNet while freezing the text encoder and VAE.

我们采用 **InstructPix2Pix + ControlNet** 的整体框架，输入包括：(1) 视频帧信息，(2) 结构控制信号（Canny 边缘），(3) 动作文本提示，用于生成目标帧。为降低训练成本，我们仅对 U-Net 与 ControlNet 施加 **LoRA** 微调，并冻结 text encoder 与 VAE。

---

## Temporal Modeling Variants (Models A–D)

We compare four strategies for incorporating temporal information:

- **Model A (Baseline):** use only the **20th frame + text** to predict the 21st frame.  
- **Model B (Trajectory Fusion in Control):** fuse Canny edges from the first 20 frames (bitwise OR) as ControlNet input, but still predict from the 20th frame + text.  
- **Model C (Original Temporal Attention):** use the first 20 frames with an original temporal attention module inserted into the U-Net.  
- **Model D (Improved Temporal Attention):** same as C, but with an improved temporal attention design (temporal positional embeddings + FFN + structured pooling).

我们对比了四种时序信息引入方式：

- **Model A（基线）：**仅使用**第20帧 + 文本**预测第21帧。  
- **Model B（融合轨迹控制信号）：**将前20帧的 Canny 边缘做 OR 融合形成轨迹图作为 ControlNet 输入，但预测仍基于第20帧 + 文本。  
- **Model C（原始 Temporal Attention）：**输入前20帧，并在 U-Net 中加入原始时序注意力模块。  
- **Model D（改进 Temporal Attention）：**在 Model C 基础上改进时序注意力结构（时序位置编码 + 前馈网络 + 结构化池化）。

---

## Experimental Setup

Training setup (unless otherwise specified): batch size **4**, learning rate **1e-4**, trained for **10 epochs**. During inference, we generate the 21st frame using **20 denoising steps**. Metrics: **PSNR** and **SSIM** (averaged over test samples).

训练设置（若无特殊说明）：batch size 为 **4**，学习率 **1e-4**，训练 **10 epochs**；推理阶段使用 **20** 次 denoising steps 生成第21帧。评估指标为 **PSNR** 与 **SSIM**（对测试样本取平均）。

---

## Results

### 1) Resolution Comparison (Models A & D)

| Model | Resolution | PSNR | SSIM |
|------|------------|------|------|
| A | 96×96  | 6.39 | 0.0808 |
| A | 128×128 | 9.05 | 0.1459 |
| D | 96×96  | 9.42 | 0.2472 |
| D | 128×128 | 8.39 | 0.3125 |

Increasing resolution substantially improves the baseline (Model A). For the temporally enhanced Model D, PSNR slightly decreases at 128×128 while SSIM increases, suggesting better structural/perceptual consistency.

提高分辨率能显著提升基线模型（Model A）的效果；对于加入时序建模的 Model D，128×128 下 PSNR 略降但 SSIM 明显上升，说明结构一致性/感知质量可能更好。

### 2) Temporal Strategy Comparison (96×96)

| Model | PSNR | SSIM |
|------|------|------|
| A | 6.39 | 0.0808 |
| B | 6.40 | 0.0811 |
| C | 8.35 | 0.1531 |
| D | 9.42 | 0.2472 |

Edge fusion alone (Model B) provides limited gains. Explicit temporal modeling (Models C/D) greatly improves both PSNR and SSIM, and the improved temporal attention (Model D) performs best.

仅增强控制信号（Model B）提升有限；引入显式时序建模（Model C/D）带来显著收益，其中改进版 Temporal Attention（Model D）效果最好。

---

## Repository Structure

Typical files/folders:

- `train.py` — training entry
- `evaluate.py` — evaluation (PSNR/SSIM)
- `preprocess.py` — preprocessing utilities
- `something_dataset.py` — dataset/subset loader
- `lora_outputs/` — LoRA checkpoints / logs (may be large)
- `metrics_output/` — quantitative results
- `comparison_output/` — qualitative comparisons

仓库结构示例：

- `train.py`：训练入口
- `evaluate.py`：评估脚本（PSNR/SSIM）
- `preprocess.py`：数据预处理
- `something_dataset.py`：数据集/子集读取
- `lora_outputs/`：LoRA 输出与日志（可能较大）
- `metrics_output/`：量化指标输出
- `comparison_output/`：可视化对比结果

---

## Quickstart

> Please adjust paths and arguments based on your environment.

1) Preprocess / build subset
```bash
python preprocess.py
2) Train / LoRA fine-tuning
```bash
python train.py
3) Evaluate
```bash
python evaluate.py
