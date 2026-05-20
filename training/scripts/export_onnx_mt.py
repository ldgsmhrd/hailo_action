"""Multi-task ResNet18 → ONNX export (5 outputs).

   입력 : [1, 7, 60, 25] (B, C, T, J)
   출력 : action_upper(6), action_lower(10), pose(9), hand(3), foot(3)
"""
import os, sys
import torch
import torch.nn as nn

sys.path.insert(0, '/home/ubuntu/safemotion/ResNet-ActionRecognition')

from configs.aica_multitask_config import CONFIG, CATEGORIES
from scripts.model_mt import build_multitask_model
from dataset.encoders import PseudoImageEncoder


class MultiTaskExportWrapper(nn.Module):
    """forward 가 tuple 반환하도록 wrap (ONNX 는 dict 출력 지원 안함)."""
    def __init__(self, model, head_order):
        super().__init__()
        self.model = model
        self.head_order = head_order

    def forward(self, x):
        out = self.model(x)
        return tuple(out[h] for h in self.head_order)


def main():
    out_path = os.path.join(CONFIG['paths']['model_dir'], 'action_resnet_mt.onnx')

    encoder = PseudoImageEncoder(order='tssi', channels=('pos', 'velocity', 'angle'))
    in_channels = encoder.num_channels

    model = build_multitask_model(heads=CATEGORIES, in_channels=in_channels,
                                  cfg=CONFIG['model'])
    ck = torch.load(os.path.join(CONFIG['paths']['model_dir'], 'best_mt.pth'),
                    map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()

    head_order = ['action_upper', 'action_lower', 'pose', 'hand', 'foot']
    wrapped = MultiTaskExportWrapper(model, head_order)
    wrapped.eval()

    T = CONFIG['frames_per_clip']
    J = 25   # TSSI
    dummy = torch.randn(1, in_channels, T, J)

    out_names = head_order

    print(f"Exporting → {out_path}")
    print(f"  input shape : (1, {in_channels}, {T}, {J})")
    print(f"  outputs     : {out_names}  dims={[CATEGORIES[h] for h in head_order]}")

    # dynamo=False : 신 exporter (torch.export) 는 kernel_shape attribute 누락 →
    # Hailo parser 에러. legacy TorchScript-based exporter 사용.
    torch.onnx.export(
        wrapped, dummy, out_path,
        input_names=['input'],
        output_names=out_names,
        opset_version=11,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"\nSaved: {out_path}  ({os.path.getsize(out_path)/1024/1024:.1f} MB)")

    # 검증
    import onnx
    m = onnx.load(out_path)
    onnx.checker.check_model(m)
    print("ONNX check OK")
    print(f"  input : {m.graph.input[0].name} → shape {[d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]}")
    for o in m.graph.output:
        print(f"  output: {o.name} → shape {[d.dim_value for d in o.type.tensor_type.shape.dim]}")


if __name__ == '__main__':
    main()
