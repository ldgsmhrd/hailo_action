"""PyTorch checkpoint → ONNX (Hailo parser 호환).

사용:
    python scripts/export_onnx.py \\
        --ckpt checkpoints/best_ntu60_mb4_xsub.pth \\
        --out psp_mb4.onnx \\
        --model mb4 \\
        --in-channels 24 \\
        --num-classes 60

주의: PyTorch 2.0+ 의 dynamo export 는 Hailo parser 가 인식 못 합니다.
본 스크립트는 자동으로 dynamo=False (legacy ONNX export) 사용.
"""
import argparse
import os
import sys
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts.train import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True, help='Output ONNX path')
    ap.add_argument('--model', default='mb4',
                    choices=['mb4', 'psp_mb4', 'mb4_2d', 'psp_mb4_2d'])
    ap.add_argument('--in-channels', type=int, default=24,
                    help='24 (MB4-3D: 3D xyz + 2body + bone motion), '
                         '16 (MB4-2D: 2D xy + 2body + bone motion)')
    ap.add_argument('--num-classes', type=int, default=60,
                    help='60 (NTU60) or 120 (NTU120)')
    ap.add_argument('--base-ch', type=int, default=64)
    ap.add_argument('--frames', type=int, default=64)
    ap.add_argument('--joints', type=int, default=25)
    ap.add_argument('--opset', type=int, default=11)
    args = ap.parse_args()

    model = build_model(args.model, args.num_classes,
                        args.in_channels, args.base_ch)
    ck = torch.load(args.ckpt, map_location='cpu')
    state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {args.ckpt}  params={n_params/1e6:.2f}M")

    dummy = torch.randn(1, args.in_channels, args.frames, args.joints)

    torch.onnx.export(
        model, dummy, args.out,
        input_names=['input'],
        output_names=['logits'],
        opset_version=args.opset,
        dynamo=False,    # ★ Hailo parser 호환 필수
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Exported: {args.out}  ({size_mb:.2f} MB)")
    print(f"Input shape: [1, {args.in_channels}, {args.frames}, {args.joints}]  (NCHW)")
    print("Hailo 컴파일 시 `--tensor-shapes 'input=[1," +
          f"{args.in_channels},{args.frames},{args.joints}]'` 로 지정.")


if __name__ == '__main__':
    main()
