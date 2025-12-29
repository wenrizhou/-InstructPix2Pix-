import torch
from torch.utils.data import DataLoader
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    ControlNetModel,
    UNet2DConditionModel
)
from transformers import AutoTokenizer
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
import os
from tqdm import tqdm
from datetime import datetime
from something_dataset import SomethingDataset

CONTROLNET_PRE = "lllyasviel/sd-controlnet-canny"
# ------------------------------------------------------------
#                 1. 训练参数
# ------------------------------------------------------------
class Args:
    npz_path = "something_subset.npz"
    pretrained_model = "dir"      # InstructPix2Pix, 可以本地下载换成路径
    controlnet_model = "lllyasviel/sd-controlnet-canny"  # ControlNet
    output_dir = "lora_outputs"
    lr = 1e-4
    batch_size = 4
    num_epochs = 10
    max_text_len = 32
    gradient_accumulation = 1
    seed = 42
    image_size = 96

def main():
    args = Args()


    # ------------------------------------------------------------
    #                 2. Accelerator
    # ------------------------------------------------------------
    accelerator = Accelerator()
    device = accelerator.device

    torch.manual_seed(args.seed)


    # ------------------------------------------------------------
    #                 3. 加载 Tokenizer
    # ------------------------------------------------------------
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.float16,
    )

    tokenizer = pipe.tokenizer


    # ------------------------------------------------------------
    #                 4. 构建 Dataset & DataLoader
    # ------------------------------------------------------------
    dataset = SomethingDataset(args.npz_path, tokenizer=tokenizer, image_size=args.image_size)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    # ------------------------------------------------------------
    #                 5. 加载基础模型：InstructPix2Pix + ControlNet
    # ------------------------------------------------------------


    # ControlNet
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model,
        torch_dtype=torch.float16,
    )

    pipe.controlnet = controlnet

    unet = pipe.unet
    pipe.unet.to(accelerator.device)
    pipe.vae.to(accelerator.device)
    pipe.controlnet.to(accelerator.device)
    pipe.text_encoder.to(accelerator.device)

    # ------------------------------------------------------------
    #      6. 注入 LoRA（只训练 UNet + ControlNet 的 LoRA）
    # ------------------------------------------------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "to_q", "to_k", "to_v",
            "ff.net.0.proj", "ff.net.2",
        ],
        lora_dropout=0.1,
        bias="none"
    )

    unet = get_peft_model(unet, lora_config)
    controlnet = get_peft_model(controlnet, lora_config)


    # ------------------------------------------------------------
    #                 7. Freeze原模型
    # ------------------------------------------------------------
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    # UNet & ControlNet 基础权重冻结
    for name, param in unet.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad_(False)

    for name, param in controlnet.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad_(False)

    # 但 LoRA 的参数允许训练
    for p in unet.parameters():
        if p.requires_grad == False and "lora" in p.__class__.__name__:
            p.requires_grad_(True)

    for p in controlnet.parameters():
        if p.requires_grad == False and "lora" in p.__class__.__name__:
            p.requires_grad_(True)


    # ------------------------------------------------------------
    #                 8. Optimizer
    # ------------------------------------------------------------
    optimizer = torch.optim.Adam(
        list(unet.parameters()) + list(controlnet.parameters()),
        lr=args.lr
    )

    # ------------------------------------------------------------
    #            9. 准备模型 for training
    # ------------------------------------------------------------
    unet, controlnet, optimizer, loader = accelerator.prepare(
        unet, controlnet, optimizer, loader
    )

    def to_device_dtype(tensor, dev, tgt_dtype):
        return tensor.to(dev, dtype=tgt_dtype)

    def encode_imgs(images):
        # images: tensor (B,3,H,W) in [0,1], float32
        # Move to device and proper dtype for VAE
        imgs = to_device_dtype(images, device, pipe.vae.dtype)
        imgs = imgs * 2.0 - 1.0
        with torch.no_grad():
            latents = pipe.vae.encode(imgs).latent_dist.sample()
            latents = latents * 0.18215
        return latents

    # ------------------------------------------------------------
    #                 10. 训练循环
    # ------------------------------------------------------------
    global_step = 0

    for epoch in range(args.num_epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            # batch expected fields: "image","target","control_image","text"
            image = batch["image"]       # (B,3,H,W) in [0,1], float32
            target = batch["target"]
            control = batch["control_image"]
            texts = batch["text"]        # may be raw strings or tokenized dict
            
            B = image.shape[0]

            # ----- Tokenize texts if they are raw strings -----
            if isinstance(texts, (list, tuple)) or (hasattr(texts, "dtype") and texts.dtype == object):
                tokenized = tokenizer(list(texts), padding="max_length", truncation=True, max_length=args.max_text_len, return_tensors="pt")
                input_ids = tokenized.input_ids.to(device)
                attn_mask = tokenized.attention_mask.to(device)
            else:
                # if dataset already returned tokenized dict
                input_ids = texts["input_ids"].to(device)
                attn_mask = texts["attention_mask"].to(device)
                if attn_mask.ndim > 2:
                    attn_mask = attn_mask.view(attn_mask.size(0), -1)

            # ----- text embeds (no grad) -----
            with torch.no_grad():
                text_embeds = pipe.text_encoder(input_ids, attention_mask=attn_mask)[0]  # (B, seq, dim)

            # ----- Encode images to latents -----
            latents_source = encode_imgs(image)   # source current frame (B,4,h,w)
            latents_target = encode_imgs(target)  # future frame (B,4,h,w)

            # ----- ControlNet conditioning: prepare control image -----
            # control comes as (B,3,H,W) in [0,1], but ControlNet usually expects image in [0,1] or normalized;
            # diffusers utilities may have helper; here we ensure dtype & device
            control = to_device_dtype(control, device, pipe.controlnet.dtype)


            # ----- Prepare noisy latents -----
            noise = torch.randn_like(latents_target)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long)

            noisy_latents = pipe.scheduler.add_noise(latents_target, noise, timesteps)

            # ----- For InstructPix2Pix we must CONCAT noisy_latents with source latents -----
            # model_input shape -> (B, 8, h, w)
            model_input = torch.cat([noisy_latents, latents_source], dim=1)

            # ----- Run ControlNet to get residuals -----
            # ControlNet forward signatures differ by versions; many return tuples (down_res, mid_res)
            # Here we call controlnet_lora directly:
            try:
                ctrl_outputs = controlnet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=text_embeds,
                    controlnet_cond=control,
                    return_dict=False
)
                # normalize returned forms
                if isinstance(ctrl_outputs, tuple) or isinstance(ctrl_outputs, list):
                    down_block_res_samples, mid_block_res_sample = ctrl_outputs
                else:
                    # if returns a dict-like, adapt below
                    down_block_res_samples = ctrl_outputs.get("down_block_res_samples", None)
                    mid_block_res_sample = ctrl_outputs.get("mid_block_res_sample", None)
            except TypeError:
                # Some versions use different arg names
                ctrl_outputs = controlnet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=text_embeds,
                    controlnet_cond=control,
                    return_dict=False
)
                down_block_res_samples, mid_block_res_sample = ctrl_outputs

            # ----- UNet forward (use LoRA unet) -----
            # UNet signature expects model_input (B,8,h,w), timesteps, encoder_hidden_states,
            # plus down_block_additional_residuals & mid_block_additional_residual
            unet_kwargs = dict(
                sample=model_input,  # some diffusers versions expect 'sample' as first positional arg
                timesteps=timesteps,
                encoder_hidden_states=text_embeds
            )

            # Many versions: unet(model_input, timesteps, encoder_hidden_states, down_block_additional_residuals=..., mid_block_additional_residual=...)
            try:
                model_output = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=text_embeds,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample
                )
                # diffusers unet returns ModelOutput with .sample
                if hasattr(model_output, "sample"):
                    noise_pred = model_output.sample
                else:
                    noise_pred = model_output
            except TypeError:
                # fallback signature
                model_output = unet(model_input, timesteps, encoder_hidden_states=text_embeds)
                if hasattr(model_output, "sample"):
                    noise_pred = model_output.sample
                else:
                    noise_pred = model_output

            # ----- Loss: predict noise (MSE) -----
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # Backprop (via accelerator)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            pbar.set_postfix({"loss": float(loss.detach().cpu())})

        # --- epoch end: save LoRA adapters ---
        if accelerator.is_main_process:
            now = datetime.now().strftime("%Y%m%d-%H%M%S")
            save_dir = os.path.join(args.output_dir, f"epoch{epoch+1}_{now}")
            os.makedirs(save_dir, exist_ok=True)
            # save LoRA adapters: unwrap models to base wrappers
            accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_dir, "unet_lora"))
            accelerator.unwrap_model(controlnet).save_pretrained(os.path.join(save_dir, "controlnet_lora"))
            print(f"Saved LoRA to {save_dir}")

if __name__ == "__main__":
    main()