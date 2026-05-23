"""PSP-Net MB4 — 단일 모델 4-Branch.

기존 MB (2-branch) 를 4-branch 로 확장.
  - Joint        (x, y, z × 2body = 6 ch)
  - Joint Motion (vx, vy, vz × 2body = 6 ch)
  - Bone         (bone_dx, dy, dz × 2body = 6 ch)
  - Bone Motion  (bone_vx, vy, vz × 2body = 6 ch)

총 24 ch 입력 → 4 stream split → 각 branch → fusion → 단일 head.

단일 ONNX / 단일 HEF / NPU 1번 추론.
파라미터 ~1.5M (MB-3D 와 비슷).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from psp_net.models.psp_mb import (
    SqueezeExcitation, BodyPartConv, STDecoupledBlock, StreamBranch
)


# 4 stream channel index (24 channel layout)
# body 0: ch [0-2]=joint, [3-5]=motion, [6-8]=bone, [9-11]=bone_motion
# body 1: ch [12-14], [15-17], [18-20], [21-23]
STREAM_INDICES_4 = {
    'joint':        [0, 1, 2, 12, 13, 14],
    'joint_motion': [3, 4, 5, 15, 16, 17],
    'bone':         [6, 7, 8, 18, 19, 20],
    'bone_motion':  [9, 10, 11, 21, 22, 23],
}


class PSPNetMB4(nn.Module):
    """4-Branch 단일 모델.

    입력: [B, 24, 64, 25]
    출력: [B, num_classes]
    """

    def __init__(self, num_classes=60, in_channels=24, base_ch=48):
        super().__init__()
        self.stream_names = list(STREAM_INDICES_4.keys())

        # 4 branch (각 mini PSP-Net)
        self.branches = nn.ModuleDict({
            name: StreamBranch(in_ch=6, base_ch=base_ch)
            for name in self.stream_names
        })
        out_ch_per_branch = base_ch * 2   # StreamBranch out

        # Fusion: 4 × out_ch_per_branch concat
        fused_ch = out_ch_per_branch * 4    # = 384 (base_ch=48 일 때)
        fusion_ch = base_ch * 4              # = 192
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_ch, fusion_ch, 1, bias=False),
            nn.BatchNorm2d(fusion_ch),
            nn.ReLU(inplace=True),
            STDecoupledBlock(fusion_ch, fusion_ch),
        )
        self.se = SqueezeExcitation(fusion_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(fusion_ch, num_classes)

        # Channel index 를 buffer 로 등록 (ONNX 호환)
        for name, idx in STREAM_INDICES_4.items():
            self.register_buffer(
                f'{name}_idx',
                torch.tensor(idx, dtype=torch.long)
            )

    def forward(self, x):
        # x: [B, 24, T, J]
        # 4 stream split
        features = []
        for name in self.stream_names:
            idx = getattr(self, f'{name}_idx')
            x_stream = x.index_select(1, idx)              # [B, 6, T, J]
            f = self.branches[name](x_stream)              # [B, base*2, T', 5]
            features.append(f)

        # Fusion
        f = torch.cat(features, dim=1)                     # [B, base*8, T', 5]
        f = self.fusion(f)                                  # [B, base*4, T', 5]
        f = self.se(f)
        f = self.pool(f).flatten(1)                        # [B, base*4]
        f = self.dropout(f)
        return self.head(f)


def build_psp_mb4(num_classes=60, in_channels=24, base_ch=48):
    return PSPNetMB4(num_classes=num_classes, in_channels=in_channels, base_ch=base_ch)


if __name__ == '__main__':
    for base in [32, 48, 64]:
        model = build_psp_mb4(num_classes=60, in_channels=24, base_ch=base)
        n = sum(p.numel() for p in model.parameters())
        print(f"base_ch={base}: params={n/1e6:.2f}M")

    print()
    model = build_psp_mb4(num_classes=60, in_channels=24, base_ch=48)
    x = torch.randn(2, 24, 64, 25)
    out = model(x)
    print(f"  base_ch=48: input={tuple(x.shape)}  output={tuple(out.shape)}")

    # ONNX export 호환 검증
    try:
        torch.onnx.export(model, x[:1], '_test_mb4.onnx',
                          input_names=['input'], output_names=['logits'],
                          opset_version=11, dynamo=False, do_constant_folding=True)
        print("  ✅ ONNX export OK")
    except Exception as e:
        print(f"  ❌ ONNX export 실패: {e}")
