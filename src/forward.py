"""YOLOv8 网络逐层手动前向传播模块。"""
import torch


class ForwardRunner:
    """按照 Ultralytics DetectionModel 中各层的 `f` 路由信息执行网络。"""

    def __init__(self, model: torch.nn.Module):
        self.model = model.eval()

    @torch.inference_mode()
    def run(self, tensor: torch.Tensor):
        x = tensor
        outputs = []
        layers = self.model.model
        print("\n[前向传播] 开始逐层执行")
        print(f"  输入 shape: {tuple(tensor.shape)}; 网络层数: {len(layers)}")

        for index, layer in enumerate(layers):
            if layer.f != -1:
                if isinstance(layer.f, int):
                    x = outputs[layer.f]
                else:
                    x = [x if source == -1 else outputs[source] for source in layer.f]
            x = layer(x)
            outputs.append(x)
            if isinstance(x, torch.Tensor):
                shape_info = str(tuple(x.shape))
            elif isinstance(x, (tuple, list)):
                shape_info = "[" + ", ".join(
                    str(tuple(item.shape)) if isinstance(item, torch.Tensor) else type(item).__name__
                    for item in x
                ) + "]"
            else:
                shape_info = type(x).__name__
            print(f"  Layer {index:02d} ({layer.__class__.__name__}): {shape_info}")

        print("[前向传播] 完成")
        self._print_output_summary(x)
        return x

    @staticmethod
    def _print_output_summary(output):
        if isinstance(output, torch.Tensor):
            print(f"  网络输出 Tensor shape: {tuple(output.shape)}")
        elif isinstance(output, (tuple, list)):
            print(f"  网络输出类型: {type(output).__name__}, 元素数: {len(output)}")
            for i, item in enumerate(output):
                print(f"    output[{i}]: shape={tuple(item.shape)}" if isinstance(item, torch.Tensor)
                      else f"    output[{i}]: type={type(item).__name__}")
        else:
            print(f"  网络输出类型: {type(output).__name__}")
