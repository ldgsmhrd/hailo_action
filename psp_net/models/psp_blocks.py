"""PSP-Net 공용 building blocks.

MB4 (3D) / MB4-2D (2D) 두 모델이 공유하는 NPU 호환 구성 요소.
NPU 표준 op 만 사용 (Conv2d, BatchNorm, ReLU, AdaptiveAvgPool, Sigmoid, Linear, Concat, Slice).

  - SqueezeExcitation : 채널 어텐션 (GAP → FC → Sigmoid → scale)
  - BodyPartConv      : 25 관절 → 5 부위 × 5 슬롯, 부위별 독립 grouped conv
  - STDecoupledBlock  : 공간(1×3) → 시간(3×1) 분리 합성곱 + residual
  - StreamBranch      : 단일 스트림 mini PSP-Net (BodyPartConv → cross 1×1 → ST blocks)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        s = self.pool(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class BodyPartConv(nn.Module):
    """부위별 독립 conv (PSP-Net 본체와 동일 패턴)."""

    def __init__(self, in_ch, out_ch, num_parts=5, joints_per_part=5, temporal_kernel=3):
        super().__init__()
        self.num_parts = num_parts
        self.joints_per_part = joints_per_part
        self.part_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=(temporal_kernel, joints_per_part),
                          padding=(temporal_kernel // 2, 0), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ) for _ in range(num_parts)
        ])

    def forward(self, x):
        outs = []
        for i, conv in enumerate(self.part_convs):
            s = i * self.joints_per_part
            e = s + self.joints_per_part
            outs.append(conv(x[:, :, :, s:e]))
        return torch.cat(outs, dim=3)


class STDecoupledBlock(nn.Module):
    """공간 (1×3) + 시간 (3×1) 분리 + residual."""

    def __init__(self, in_ch, out_ch, temporal_kernel=3, spatial_kernel=3):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, spatial_kernel),
                      padding=(0, spatial_kernel // 2), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (temporal_kernel, 1),
                      padding=(temporal_kernel // 2, 0), bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.residual = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        res = self.residual(x)
        x = self.spatial(x)
        x = self.temporal(x)
        return F.relu(x + res, inplace=True)


class StreamBranch(nn.Module):
    """단일 stream 의 mini PSP-Net.

    [B, in_ch, T=64, J=25] → [B, out_ch, T'=16, J'=5]
    """

    def __init__(self, in_ch, base_ch=64):
        super().__init__()
        # 부위별 독립 conv (joint 25 → 5 부위)
        self.body_part = BodyPartConv(in_ch, base_ch,
                                       num_parts=5, joints_per_part=5)
        # cross-part 1x1
        self.cross = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        # S-T blocks + temporal pooling
        self.st1 = STDecoupledBlock(base_ch, base_ch * 2)
        self.st2 = STDecoupledBlock(base_ch * 2, base_ch * 2)
        # temporal downsample 안에 들어가 있음 (avg_pool)
        self.out_ch = base_ch * 2

    def forward(self, x):
        # x: [B, in_ch, 64, 25]
        x = self.body_part(x)                       # [B, base, 64, 5]
        x = self.cross(x)                           # [B, base, 64, 5]
        x = self.st1(x)
        x = F.avg_pool2d(x, (2, 1))                  # [B, base*2, 32, 5]
        x = self.st2(x)
        x = F.avg_pool2d(x, (2, 1))                  # [B, base*2, 16, 5]
        return x
