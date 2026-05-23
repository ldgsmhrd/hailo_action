"""PSP-Net NTU60 버전 — 단일 head 60-class.

기존 SmartNVR psp_net (5 head, 7ch) 와 분리. 완전 독립 모듈.

입력: [B, 6, T=64, J=25]  (NTU 25 joint, body-part reorder 후)
       채널: x, y, vx, vy, bone_dx, bone_dy  (z 는 drop — YOLO-Pose 정합성)
출력: [B, 60]  logits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        s = self.pool(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class BodyPartConv(nn.Module):
    """5 부위 독립 conv. joint 25 → 5 (부위당 1슬롯)."""

    def __init__(self, in_ch, out_ch, num_parts=5, joints_per_part=5, temporal_kernel=3):
        super().__init__()
        self.num_parts = num_parts
        self.joints_per_part = joints_per_part
        self.part_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=(temporal_kernel, joints_per_part),
                          padding=(temporal_kernel // 2, 0),
                          bias=False),
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
    """공간 (1×3) + 시간 (3×1) 분리 conv + residual."""

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


class MultiScaleTemporal(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 2, 4, 8)):
        super().__init__()
        n = len(dilations)
        assert out_ch % n == 0
        per = out_ch // n
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, per, (3, 1),
                          padding=(d, 0), dilation=(d, 1), bias=False),
                nn.BatchNorm2d(per),
                nn.ReLU(inplace=True),
            ) for d in dilations
        ])

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class PSPNetNTU(nn.Module):
    """NTU60 단일 head 버전. 60 class 출력.

    [B, 6, 64, 25] → [B, 60]

    파라미터: 약 1.08M (SmartNVR 5-head 버전 1.07M 과 비슷)
    """

    def __init__(self, num_classes=60, in_channels=6,
                 num_parts=5, joints_per_part=5):
        super().__init__()
        self.num_parts = num_parts
        self.joints_per_part = joints_per_part

        self.body_part = BodyPartConv(in_channels, 64,
                                       num_parts=num_parts,
                                       joints_per_part=joints_per_part)
        self.cross_part = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.st_block1 = STDecoupledBlock(64, 128)
        self.st_block2 = STDecoupledBlock(128, 256)
        self.st_block3 = STDecoupledBlock(256, 256)

        self.ms_temporal = MultiScaleTemporal(256, 256)
        self.se = SqueezeExcitation(256)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, 6, 64, 25]
        x = self.body_part(x)              # [B, 64, 64, 5]
        x = self.cross_part(x)             # [B, 64, 64, 5]

        x = self.st_block1(x)              # [B, 128, 64, 5]
        x = F.avg_pool2d(x, (2, 1))         # [B, 128, 32, 5]

        x = self.st_block2(x)              # [B, 256, 32, 5]
        x = F.avg_pool2d(x, (2, 1))         # [B, 256, 16, 5]

        x = self.st_block3(x)              # [B, 256, 16, 5]
        x = self.ms_temporal(x)            # [B, 256, 16, 5]
        x = self.se(x)                     # [B, 256, 16, 5]

        x = self.pool(x).flatten(1)        # [B, 256]
        x = self.dropout(x)
        return self.head(x)                 # [B, 60]


def build_psp_net_ntu(num_classes=60, in_channels=6):
    return PSPNetNTU(num_classes=num_classes, in_channels=in_channels)


if __name__ == '__main__':
    model = build_psp_net_ntu(num_classes=60, in_channels=6)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PSP-Net NTU 파라미터: {n_params/1e6:.2f}M")
    x = torch.randn(4, 6, 64, 25)
    out = model(x)
    print(f"  input  {tuple(x.shape)}")
    print(f"  output {tuple(out.shape)}")
