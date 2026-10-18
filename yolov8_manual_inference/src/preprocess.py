"""YOLOv8 手动推理前处理模块。"""
from dataclasses import dataclass

import cv2
import numpy as np
import torch

@dataclass
class ImageMeta:
    original_h: int
    original_w: int
    scale: float
    pad_left: int
    pad_top: int

class Preprocessor:
    """将 OpenCV BGR 图片转换为模型所需的 RGB BCHW Tensor。"""

    def __init__(self, input_size: int = 640, device: torch.device | str = "cpu"):
        if input_size <= 0:
            raise ValueError("input_size 必须大于 0")
        self.input_size = int(input_size)
        self.device = torch.device(device)

    def process(self, image: np.ndarray) -> tuple[torch.Tensor, ImageMeta]:
        if image is None or not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("输入必须是有效的 H×W×3 彩色图像")

        h0, w0 = image.shape[:2]
        scale = min(self.input_size / w0, self.input_size / h0)
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw, dh = self.input_size - new_w, self.input_size - new_h
        left, top = dw // 2, dh // 2
        padded = cv2.copyMakeBorder(
            resized, top, dh - top, left, dw - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        tensor = torch.from_numpy(chw).float().div_(255.0).unsqueeze(0).to(self.device)
        meta = ImageMeta(h0, w0, scale, left, top)

        print("\n[前处理] 完成")
        print(f"  原图尺寸 (H, W): ({h0}, {w0})")
        print(f"  缩放比例: {scale:.6f}; padding (left, top): ({left}, {top})")
        print(f"  Tensor shape: {tuple(tensor.shape)}; dtype: {tensor.dtype}; device: {tensor.device}")
        print(f"  Tensor 数值范围: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
        return tensor, meta
