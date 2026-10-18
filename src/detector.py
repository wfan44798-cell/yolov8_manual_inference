"""YOLOv8 手动推理的统一协调入口。"""
import cv2
import numpy as np
import torch

from .forward import ForwardRunner
from .postprocess import Postprocessor
from .preprocess import Preprocessor


class YOLODetector:
    """把前处理、前向传播和后处理组合成一个可复用的检测器对象。"""

    def __init__(self, model: torch.nn.Module, input_size: int = 640,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 device: str = None, class_names=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.preprocessor = Preprocessor(input_size=input_size, device=self.device)
        self.forward_runner = ForwardRunner(self.model)
        self.postprocessor = Postprocessor(conf_threshold, iou_threshold, class_names)

    def preprocess(self, image: np.ndarray):
        return self.preprocessor.process(image)

    def forward(self, tensor: torch.Tensor):
        return self.forward_runner.run(tensor)

    def postprocess(self, output, meta):
        return self.postprocessor.process(output, meta)

    def detect(self, image: np.ndarray) -> list[dict]:
        """统一调用入口：image -> preprocess -> forward -> postprocess。"""
        print("\n========== YOLODetector.detect 开始 ==========")
        tensor, meta = self.preprocess(image)
        output = self.forward(tensor)
        results = self.postprocess(output, meta)
        print("========== YOLODetector.detect 结束 ==========")
        return results


def draw_detections(image: np.ndarray, detections) -> np.ndarray:
    """在原图上绘制检测框。"""
    result = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    return result
