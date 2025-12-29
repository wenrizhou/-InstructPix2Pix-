# AI Model for Human-Object Interaction Frame Prediction

This repository explores **video frame prediction** on the **Something-Something V2** dataset: generating the **21st frame** conditioned on the first 20 frames and the action text label.  
Our framework adapts **InstructPix2Pix** as the diffusion backbone and integrates **ControlNet** for structural constraints, with **LoRA** fine-tuning for parameter-efficient training. We further introduce an **improved Temporal Attention** module to better capture multi-frame temporal dependencies. :contentReference[oaicite:1]{index=1}

## Highlights
- **Task:** predict the next frame (21st) in human-object interaction videos using visual + textual conditioning. :contentReference[oaicite:2]{index=2}  
- **Backbone:** InstructPix2Pix (UNet + Text Encoder + VAE). :contentReference[oaicite:3]{index=3}  
- **Structure control:** ControlNet with **Canny edge maps** as spatial constraints. :contentReference[oaicite:4]{index=4}  
- **Efficient training:** LoRA applied to UNet and ControlNet; text encoder and VAE are frozen. :contentReference[oaicite:5]{index=5}  
- **Temporal modeling:** Temporal Attention inserted into UNet, operating on VAE latents from the first 20 frames; includes learnable temporal positional embeddings + feed-forward block + structured pooling. :contentReference[oaicite:6]{index=6}  

---

## Methods

We compare four strategies for incorporating temporal information: :contentReference[oaicite:7]{index=7}

- **Model A (Baseline):** use only the **20th frame + text** to predict the 21st frame.  
- **Model B (Edge Aggregation):** fuse temporal info into ControlNet input by aggregating Canny edges from the first 20 frames, still predicting from the 20th frame + text.  
- **Model C (Original Temporal Attention):** input the first 20 frames and insert the original temporal attention module into UNet.  
- **Model D (Improved Temporal Attention):** same as C but with an improved temporal attention design (positional embedding + FFN + structured pooling). :contentReference[oaicite:8]{index=8}  

---

## Experimental Setup

- **Dataset:** selected subsets from Something-Something V2 (e.g., *cover*, *drop*, *move*). :contentReference[oaicite:9]{index=9}  
- **Resolution:** compare **96×96** and **128×128**. :contentReference[oaicite:10]{index=10}  
- **Training:** batch size = 4, learning rate = 1e-4, epochs = 10. :contentReference[oaicite:11]{index=11}  
- **Inference:** 20 denoising steps. :contentReference[oaicite:12]{index=12}  
- **Metrics:** PSNR and SSIM (average over test samples). :contentReference[oaicite:13]{index=13}  

---

## Results

### 1) Effect of Input Resolution (Models A & D)
| Model | Resolution | PSNR | SSIM |
|------|------------|------|------|
| A | 96×96  | 6.39 | 0.0808 |
| A | 128×128 | 9.05 | 0.1459 |
| D | 96×96  | 9.42 | 0.2472 |
| D | 128×128 | 8.39 | 0.3125 |

Key observations:
- Higher resolution improves the baseline (Model A) significantly.
- For temporally enhanced Model D, PSNR may drop slightly at 128×128 while SSIM increases, suggesting better structural/perceptual consistency. :contentReference[oaicite:14]{index=14}  

### 2) Temporal Modeling Strategy Comparison (96×96)
| Model | PSNR | SSIM |
|------|------|------|
| A | 6.39 | 0.0808 |
| B | 6.40 | 0.0811 |
| C | 8.35 | 0.1531 |
| D | 9.42 | 0.2472 |

Key observations:
- Edge aggregation alone (Model B) brings limited gains over baseline.
- Explicit temporal modeling (Models C/D) yields large improvements.
- The improved temporal attention (Model D) achieves the best overall performance. :contentReference[oaicite:15]{index=15}  

---

## Repository Structure (example)

> Adjust this section if your folder names differ.

- `train.py` — training entry
- `evaluate.py` — evaluation script (PSNR/SSIM)
- `preprocess.py` — preprocessing utilities
- `something_dataset.py` — dataset loading / sampling utilities
- `lora_outputs/` — LoRA checkpoints / logs (may be large)
- `metrics_output/` — metric results
- `comparison_output/` — qualitative comparisons / visual outputs

---

## How to Run

> The exact commands depend on your environment and paths. A typical workflow:

1. **Preprocess / prepare subset**
   ```bash
   python preprocess.py
2. **Train**
   python train.py
3. **Evaluate**
   python evaluate.py
