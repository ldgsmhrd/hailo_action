"""PSP-Net MB4-2D — 2D 입력 단일 모델 4-Branch.

MB4 (3D, 24ch) 의 2D 입력 버전.
  - Joint        (x, y × 2body = 4 ch)
  - Joint Motion (vx, vy × 2body = 4 ch)
  - Bone         (bx, by × 2body = 4 ch)
  - Bone Motion  (bvx, bvy × 2body = 4 ch)

총 16 ch 입력 → 4 stream split → 각 branch → fusion → 단일 head.

3D MB4 와 동일한 구조 (4-branch + 1×1 fusion + STDecoupled + SE) 이지만
입력 채널만 16 으로 축소.  파라미터 ~1.4M.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from psp_net.models.psp_blocks import (
    SqueezeExcitation, BodyPartConv, STDecoupledBlock, StreamBranch
)


# 4 stream channel index (16 channel layout)
# body 0: ch [0-1]=joint xy, [2-3]=motion xy, [4-5]=bone xy, [6-7]=bone_motion xy
# body 1: ch [8-9],  [10-11], [12-13], [14-15]
STREAM_INDICES_4_2D = {
    'joint':        [0, 1,  8,  9],
    'joint_motion': [2, 3, 10, 11],
    'bone':         [4, 5, 12, 13],
    'bone_motion':  [6, 7, 14, 15],
}


class PSPNetMB4_2D(nn.Module):
    """4-Branch 단일 모델 (2D 입력).

    입력: [B, 16, 64, 25]
    출력: [B, num_classes]
    """

    def __init__(self, num_classes=60, in_channels=16, base_ch=48):
        super().__init__()
        self.stream_names = list(STREAM_INDICES_4_2D.keys())

        # 4 branch (각 mini PSP-Net) — stream 당 in_ch=4
        self.branches = nn.ModuleDict({
            name: StreamBranch(in_ch=4, base_ch=base_ch)
            for name in self.stream_names
        })
        out_ch_per_branch = base_ch * 2

        fused_ch = out_ch_per_branch * 4    # = 384 (base_ch=48)
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
        for name, idx in STREAM_INDICES_4_2D.items():
            self.register_buffer(
                f'{name}_idx',
                torch.tensor(idx, dtype=torch.long)
            )

    def forward(self, x):
        # x: [B, 16, T, J]
        features = []
        for name in self.stream_names:
            idx = getattr(self, f'{name}_idx')
            x_stream = x.index_select(1, idx)              # [B, 4, T, J]
            f = self.branches[name](x_stream)              # [B, base*2, T', 5]
            features.append(f)

        f = torch.cat(features, dim=1)                     # [B, base*8, T', 5]
        f = self.fusion(f)                                  # [B, base*4, T', 5]
        f = self.se(f)
        f = self.pool(f).flatten(1)                        # [B, base*4]
        f = self.dropout(f)
        return self.head(f)


def build_psp_mb4_2d(num_classes=60, in_channels=16, base_ch=48):
    return PSPNetMB4_2D(num_classes=num_classes, in_channels=in_channels, base_ch=base_ch)


if __name__ == '__main__':
    for base in [32, 48, 64]:
        model = build_psp_mb4_2d(num_classes=60, in_channels=16, base_ch=base)
        n = sum(p.numel() for p in model.parameters())
        print(f"base_ch={base}: params={n/1e6:.2f}M")

    print()
    model = build_psp_mb4_2d(num_classes=60, in_channels=16, base_ch=64)
    x = torch.randn(2, 16, 64, 25)
    out = model(x)
    print(f"  base_ch=64: input={tuple(x.shape)}  output={tuple(out.shape)}")

    try:
        torch.onnx.export(model, x[:1], '_test_mb4_2d.onnx',
                          input_names=['input'], output_names=['logits'],
                          opset_version=11, dynamo=False, do_constant_folding=True)
        print("  ✅ ONNX export OK")
    except Exception as e:
        print(f"  ❌ ONNX export 실패: {e}")
