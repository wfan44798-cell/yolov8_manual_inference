"""YOLOv8 检测输出后处理模块（手写 IoU 与 NMS）。"""
import torch

from .preprocess import ImageMeta


class Postprocessor:
    def __init__(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45, class_names=None):
        if not 0 <= conf_threshold <= 1:
            raise ValueError("conf_threshold 应在 [0, 1] 范围内")
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold 应在 [0, 1] 范围内")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []

    @staticmethod
    def get_raw_prediction(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            pred = output
        elif isinstance(output, (tuple, list)):
            pred = next((item for item in output if isinstance(item, torch.Tensor)), None)
            if pred is None:
                raise TypeError("模型输出 tuple/list 中没有 Tensor")
        elif isinstance(output, dict):
            pred = next((item for item in output.values() if isinstance(item, torch.Tensor)), None)
            if pred is None:
                raise TypeError("模型输出 dict 中没有 Tensor")
        else:
            raise TypeError(f"不支持的模型输出类型：{type(output)}")

        if pred.ndim != 3:
            raise ValueError(f"期望原始检测输出为 3 维 Tensor，实际 shape={tuple(pred.shape)}")
        # YOLOv8 常见布局 [B, 4+nc, N]，转成 [B, N, 4+nc]
        if pred.shape[1] < pred.shape[2]:
            pred = pred.transpose(1, 2)
        return pred

    @staticmethod
    def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        result = torch.empty_like(boxes)
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return result

    @staticmethod
    def iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        lt = torch.maximum(box[:2], boxes[:, :2])
        rb = torch.minimum(box[2:], boxes[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
        area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        return inter / (area1 + area2 - inter + 1e-7)

    def nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        order = scores.argsort(descending=True)
        keep = []
        while order.numel() > 0:
            best = order[0]
            keep.append(best)
            if order.numel() == 1:
                break
            rest = order[1:]
            ious = self.iou_one_to_many(boxes[best], boxes[rest])
            order = rest[ious <= self.iou_threshold]
        if not keep:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        return torch.stack(keep)

    def process(self, output, meta: ImageMeta) -> list[dict]:
        pred = self.get_raw_prediction(output)[0]
        if pred.shape[-1] < 5:
            raise ValueError(f"类别数不正确，预测 shape={tuple(pred.shape)}")

        boxes = pred[:, :4]
        class_scores = pred[:, 4:]
        scores, class_ids = class_scores.max(dim=1)
        mask = scores >= self.conf_threshold
        print("\n[后处理] 开始")
        print(f"  原始候选框数量: {pred.shape[0]}")
        print(f"  置信度阈值: {self.conf_threshold}; IoU 阈值: {self.iou_threshold}")
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
        print(f"  置信度筛选后: {boxes.shape[0]}")
        if boxes.numel() == 0:
            print("  没有候选框通过置信度筛选，检测结果为空")
            return []

        boxes = self.xywh_to_xyxy(boxes)
        kept = []
        for cid in class_ids.unique():
            idx = torch.where(class_ids == cid)[0]
            kept.append(idx[self.nms(boxes[idx], scores[idx])])
        keep = torch.cat(kept)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes, scores, class_ids = boxes[keep].clone(), scores[keep], class_ids[keep]
        print(f"  NMS 后保留: {len(keep)}")

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta.pad_left) / meta.scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta.pad_top) / meta.scale
        boxes[:, [0, 2]].clamp_(0, meta.original_w)
        boxes[:, [1, 3]].clamp_(0, meta.original_h)

        results = []
        for box, score, cid in zip(boxes.cpu(), scores.cpu(), class_ids.cpu()):
            class_id = int(cid.item())
            if isinstance(self.class_names, dict):
                name = self.class_names.get(class_id, str(class_id))
            else:
                name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            results.append({"bbox": [float(v) for v in box.tolist()], "confidence": float(score.item()),
                            "class_id": class_id, "class_name": name})
        print(f"[后处理] 完成，最终检测数量: {len(results)}")
        for i, item in enumerate(results):
            print(f"  Detection {i + 1}: class={item['class_name']} (id={item['class_id']}), "
                  f"confidence={item['confidence']:.4f}, bbox={item['bbox']}")
        return results
