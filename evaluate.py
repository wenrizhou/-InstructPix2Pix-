import os
import torch
from tqdm import tqdm
from diffusers import StableDiffusionInstructPix2PixPipeline, ControlNetModel
from transformers import AutoTokenizer
from something_dataset import SomethingDataset  # 你之前的数据集类
from accelerate import Accelerator
from peft import PeftModel
from PIL import Image

# -------------------------------
# 配置
# -------------------------------
class Args:
    npz_path = "something_subset.npz"           # 数据文件
    pretrained_model = "dir"                    # 原始InstructPix2Pix模型路径
    controlnet_model = "lllyasviel/sd-controlnet-canny"
    lora_unet = "lora_outputs/epoch10_20251212-024338/unet_lora"       # LoRA训练结果
    lora_controlnet = "lora_outputs/epoch10_20251212-024338/controlnet_lora"
    output_dir = "eval_results"
    batch_size = 4
    image_size = 96
    max_text_len = 32
    seed = 42
    num_inference_steps = 20                     # 生成步数，可调

args = Args()
os.makedirs(args.output_dir, exist_ok=True)
torch.manual_seed(args.seed)
accelerator = Accelerator()
device = accelerator.device

# -------------------------------
# 加载 pipeline + LoRA
# -------------------------------
print("Loading InstructPix2Pix pipeline...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    args.pretrained_model,
    torch_dtype=torch.float16,
)
pipe.to(device)
pipe.enable_attention_slicing()  # 如果显存紧张

# ControlNet
controlnet = ControlNetModel.from_pretrained(
    args.controlnet_model,
    torch_dtype=torch.float16,
)
controlnet.to(device)

# 注入LoRA
pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora_unet)
controlnet = PeftModel.from_pretrained(controlnet, args.lora_controlnet)
pipe.controlnet = controlnet

tokenizer = pipe.tokenizer

# -------------------------------
# Dataset & DataLoader
# -------------------------------
dataset = SomethingDataset(args.npz_path, tokenizer=tokenizer, image_size=args.image_size)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

# -------------------------------
# 评估循环
# -------------------------------
pipe.unet.eval()
pipe.controlnet.eval()
cnt = 0
print("Start evaluation...")
for i, batch in enumerate(tqdm(loader)):
    cnt += batch["image"].shape[0]
    images = batch["image"]          # 第一帧
    control_images = batch["control_image"]
    texts = batch["text"]

    for j in range(images.shape[0]):
        img = images[j].unsqueeze(0).to(device)            # [1,3,H,W]
        ctrl = control_images[j].unsqueeze(0).to(device)
        # 处理文本
        if isinstance(texts, (list, tuple)):
            prompt = texts[j]
        else:
            prompt = texts.get("raw_text", [str(i) for i in range(len(images))])[j]

        with torch.no_grad():
            generated = pipe(
                prompt=prompt,
                image=img,
                control_image=ctrl,
                num_inference_steps=args.num_inference_steps
            ).images[0]  # PIL Image

        # 保存结果
        save_path = os.path.join(args.output_dir, f"sample_{i*args.batch_size+j:04d}.png")
        generated.save(save_path)
    if cnt >= 20:
        break

print("Evaluation finished. Results saved to:", args.output_dir)
