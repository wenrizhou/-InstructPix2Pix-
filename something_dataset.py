import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

def get_canny_edges(img_tensor):
    """
    img_tensor: torch.Tensor [3,H,W], in [0,1]
    return: torch.Tensor [3,H,W], in [0,1]
    """
    # 转为 HWC 的 numpy
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # [H,W,3]

    # 转 uint8
    img_uint8 = (img * 255).astype(np.uint8)

    # Canny usually done on grayscale
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    # Canny
    edges = cv2.Canny(gray, 100, 200)  # 结果为 [H,W]

    # 扩展为 3 通道
    edges_3c = np.stack([edges] * 3, axis=-1)  # [H,W,3]

    # 回到 tensor 格式 [3,H,W] 并归一化
    edges_3c = torch.from_numpy(edges_3c).float() / 255.0
    edges_3c = edges_3c.permute(2, 0, 1)

    return edges_3c

class SomethingDataset(Dataset):
    def __init__(self, npz_path, tokenizer=None, image_size=96):
        """
        npz_path: 预处理后保存的 npz 文件路径
        tokenizer: 文本 tokenizer（可为 None，则返回原始字符串）
        image_size: 图像大小（保持与预处理一致）
        """
        data = np.load(npz_path, allow_pickle=True)

        self.first_frames = data["first_frame"]      # (N, H, W, 3) RGB
        self.future_frames = data["future_frame"]    # (N, H, W, 3)
        self.texts = data["text"]                    # (N,)

        self.tokenizer = tokenizer

        # 转 Tensor 和归一化的 transform
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),      # HWC → CHW, [0,255] → [0,1]
        ])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # ---------- 图像 ----------
        img_now = self.first_frames[idx]
        img_future = self.future_frames[idx]

        img_now = self.img_transform(img_now)          # → Tensor
        img_future = self.img_transform(img_future)    # → Tensor
        control_image = get_canny_edges(img_now)      # Canny 边缘图

        # ---------- 文本 ----------
        text = str(self.texts[idx])

        if self.tokenizer is not None:
            # 例如：返回 input_ids / attention_mask
            text_tokenized = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=32,
                return_tensors="pt"
            )
        else:
            text_tokenized = text   # 原始字符串

        return {
            "image": img_now,              # 输入图像 (3,H,W)
            "text": text_tokenized,        # tokenized 或原始字符串
            "control_image": control_image,      # Canny 边缘图 (3,H,W)
            "target": img_future           # 未来帧 (3,H,W)
        }
