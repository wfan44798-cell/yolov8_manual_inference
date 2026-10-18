# PROJECT STATUS

## 当前阶段

已完成第一轮推理流程模块化重构，目标是尽量保留原有目录和执行方式。

## 已完成

- 保留 `main.py`、`src/`、`data/`、`weights/`、`outputs/` 等原有目录。
- 新增 `src/preprocess.py`：封装 Letterbox、BGR→RGB、HWC→CHW、归一化和 Tensor 创建。
- 新增 `src/forward.py`：按 Ultralytics 模型层的 `f` 路由信息执行逐层前向传播。
- 新增 `src/postprocess.py`：封装预测解析、置信度筛选、xywh→xyxy、手写 IoU/NMS 和坐标还原。
- 重构 `src/detector.py`：通过 `YOLODetector` 组合调用三个阶段，保留 `draw_detections()`。
- 重构 `main.py`：通过检测器对象完成推理，并统一使用基于项目根目录的绝对路径。
- 每个阶段加入关键中间结果打印。
- 更新 README，记录目录结构、模块职责和运行方式。

## 当前可运行性

- 已完成 `python -m compileall -q main.py scr` 静态语法编译检查。
- 已用合成预测 Tensor 验证前处理和后处理 smoke test：Tensor 形状、置信度筛选、NMS、坐标还原与检测结果打印均通过。
- 当前执行环境未安装 `ultralytics`，因此尚不能在此环境完整运行 `python main.py`；需安装 `requirements.txt` 依赖后验证真实模型的逐层前向传播。
- 运行成功后应生成 `outputs/result.jpg`。

## 已知风险 / 待验证

- Ultralytics 不同版本可能改变 `DetectionModel` 层结构或检测头输出形式。
- 逐层前向传播打印每一层的 shape，日志会比较长；这是当前阶段为方便学习和调试而保留的行为。
- 当前仅针对普通目标检测任务，不支持分割、姿态估计和 OBB。
- 手动后处理未实现官方推理器全部参数（例如 max_det），输出可能与官方预测不同。

## 下一步

1. 在目标 Python 环境运行 `python main.py`，检查三阶段日志与结果图片。
2. 对照 `YOLO.predict()` 结果，排查差异并验证 NMS、坐标映射和类别名。
3. 为三个模块增加小型单元测试（尺寸转换、IoU/NMS、空检测结果）。
4. 如日志过长，可为逐层 shape 打印增加 `verbose` 开关。
5. 确认运行稳定后再考虑进一步优化目录或增加配置文件。
