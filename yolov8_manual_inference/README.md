# YOLOv8 手动推理示例

本项目使用 Ultralytics 提供的 YOLOv8 模型结构和预训练权重，但将检测推理拆分为独立的前处理、逐层前向传播和后处理模块；不调用 `YOLO.predict()` 或 `torchvision.ops.nms()`。

## 项目结构

```text
yolov8_manual_inference/
├── main.py                  # 程序入口：创建并调用 YOLODetector
├── src/
│   ├── __init__.py
│   ├── detector.py          # 流程协调器 + 绘图函数
│   ├── preprocess.py        # 缩放、Letterbox、颜色/维度转换、归一化
│   ├── forward.py           # 根据每层的 f 路由信息手动执行网络
│   └── postprocess.py       # 置信度筛选、坐标转换、手写 IoU/NMS、坐标还原
├── data/
│   └── test.jpg             # 测试图片
├── weights/
│   └── yolov8n.pt           # 模型权重
├── outputs/                 # 输出目录，运行时自动创建
├── requirements.txt
└── PROJECT_STATUS.md        # 项目进度与后续计划
```

## 环境与运行

建议使用 Python 3.10 或 3.11。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

运行路径以 `main.py` 所在目录为基准，不依赖终端当前工作目录。默认读取 `data/test.jpg` 和 `weights/yolov8n.pt`，输出到 `outputs/result.jpg`。

## 模块调用关系

```text
main.py
  └── YOLODetector.detect(image)
        ├── Preprocessor.process(image)       -> Tensor, ImageMeta
        ├── ForwardRunner.run(tensor)          -> 网络原始输出
        └── Postprocessor.process(output, meta)-> 检测结果列表
```

每个阶段会打印输入/输出尺寸和关键中间结果；后处理会打印候选框数量、置信度筛选数量、NMS 保留数量及最终检测列表。

## 说明与限制

- 适用于 Ultralytics YOLOv8 普通目标检测（detect）模型，不适用于 segment、pose、obb 等任务。
- 代码假设检测头原始输出包含 `[B, 4 + nc, N]` 形式的预测 Tensor。Ultralytics 版本不同可能导致输出结构变化，需要根据实际输出调整解析逻辑。
- 这是用于学习的手动推理流程，不覆盖官方推理器的所有细节，结果可能与 `YOLO.predict()` 有差异。
