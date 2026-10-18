"""项目入口：加载模型、创建检测器对象并执行一次检测。"""
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.detector import YOLODetector, draw_detections


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    weights_path = PROJECT_ROOT / "weights" / "yolov8n.pt"
    image_path = PROJECT_ROOT / "data" / "test.jpg"
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.is_file():
        raise FileNotFoundError(f"找不到模型权重：{weights_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到测试图片：{image_path}")

    print(f"加载模型: {weights_path}")
    yolo = YOLO(str(weights_path))
    detector = YOLODetector(
        model=yolo.model,
        input_size=640,
        conf_threshold=0.3,
        iou_threshold=0.45,
        class_names=yolo.names,
    )

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"OpenCV 无法读取图片：{image_path}")

    # 通过 detector 对象调用完整流程；每个阶段会打印中间结果。
    detections = detector.detect(image)
    visualized = draw_detections(image, detections)
    output_path = output_dir / "result.jpg"
    if not cv2.imwrite(str(output_path), visualized):
        raise RuntimeError(f"图片保存失败：{output_path}")

    print(f"\n检测完成：共检测到 {len(detections)} 个目标")
    print(f"结果图片保存到：{output_path}")


if __name__ == "__main__":
    main()
