"""PSP-Net: Partitioned Skeletal Pseudo-image Network

5가지 NPU 호환 컴포넌트의 조합:
  ① Body-Part Block Pseudo-image (joint axis 재배열)
  ② Body-Part Conv (각 부위 5 joint 독립 처리)
  ③ Spatial-Temporal Decoupled Conv
  ④ Multi-Scale Temporal Conv (dilation 1, 2, 4, 8)
  ⑤ Squeeze-Excitation (NPU 호환 채널 attention)
  ⑥ 5-head 분류 (action_upper / lower / pose / hand / foot)

입력: [B, 7, T=60, J=25]   (J = 5 신체부위 × 5 슬롯)
출력: dict {head_name: logits}

NPU 호환 op 만 사용 (Conv2D, BN, ReLU, AvgPool, AdaptiveAvgPool, Sigmoid, Linear).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Channel attention (Hu et al. 2018) — NPU 완전 호환."""

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
    """① + ② 신체부위별 독립 conv.

    5 신체부위 그룹 (각 5 슬롯) 을 독립 conv 로 처리.
    joint 축 축소: 25 → 5 (각 부위당 1 슬롯으로 압축).
    """

    def __init__(self, in_ch, out_ch, num_parts=5, joints_per_part=5, temporal_kernel=3):
        super().__init__()
        self.num_parts = num_parts
        self.joints_per_part = joints_per_part
        # 각 부위마다 (in_ch, T, joints_per_part) → (out_ch, T, 1) conv
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
        # x: [B, in_ch, T, num_parts*joints_per_part]
        outs = []
        for i, conv in enumerate(self.part_convs):
            start = i * self.joints_per_part
            end = start + self.joints_per_part
            outs.append(conv(x[:, :, :, start:end]))   # [B, out_ch, T, 1]
        return torch.cat(outs, dim=3)                  # [B, out_ch, T, num_parts]


class STDecoupledBlock(nn.Module):
    """③ Spatial-Temporal Decoupled Conv Block.

    spatial conv (kernel=1×3) — 부위 축 (5개) 간 mixing
    temporal conv (kernel=3×1) — 시간 축
    + residual.
    """

    def __init__(self, in_ch, out_ch, temporal_kernel=3, spatial_kernel=3):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=(1, spatial_kernel),
                      padding=(0, spatial_kernel // 2),
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv2d(out_ch, out_ch,
                      kernel_size=(temporal_kernel, 1),
                      padding=(temporal_kernel // 2, 0),
                      bias=False),
            nn.BatchNorm2d(out_ch),
        )
        if in_ch == out_ch:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.spatial(x)
        x = self.temporal(x)
        return F.relu(x + res, inplace=True)


class MultiScaleTemporal(nn.Module):
    """④ Multi-scale temporal — dilation 1, 2, 4, 8 병렬 → concat."""

    def __init__(self, in_ch, out_ch, dilations=(1, 2, 4, 8)):
        super().__init__()
        n = len(dilations)
        assert out_ch % n == 0, f"out_ch ({out_ch}) must be divisible by {n}"
        per = out_ch // n
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, per,
                          kernel_size=(3, 1),
                          padding=(d, 0),
                          dilation=(d, 1),
                          bias=False),
                nn.BatchNorm2d(per),
                nn.ReLU(inplace=True),
            ) for d in dilations
        ])

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class PSPNet(nn.Module):
    """Partitioned Skeletal Pseudo-image Network.

    파라미터: 약 1.5M (ResNet18 11.2M 대비 1/7)

    구조:
        [B, 7, 60, 25]
            ↓ BodyPartConv (in=7, out=64, parts=5)
        [B, 64, 60, 5]
            ↓ STDecoupledBlock × 1 + Pool(2,1)
        [B, 128, 30, 5]
            ↓ STDecoupledBlock × 1 + Pool(2,1)
        [B, 256, 15, 5]
            ↓ STDecoupledBlock × 1
        [B, 256, 15, 5]
            ↓ MultiScaleTemporal (dilation 1,2,4,8)
        [B, 256, 15, 5]
            ↓ SqueezeExcitation
        [B, 256, 15, 5]
            ↓ AdaptiveAvgPool2d(1)
        [B, 256, 1, 1] → flatten [B, 256]
            ↓ 5 head Linear
        {head_name: logits}
    """

    def __init__(self, heads, in_channels=7, num_parts=5, joints_per_part=5):
        super().__init__()
        self.heads_info = heads
        self.num_parts = num_parts
        self.joints_per_part = joints_per_part

        # Stage 1: Body-part conv (joint 25 → 5)
        self.body_part = BodyPartConv(in_channels, 64,
                                       num_parts=num_parts,
                                       joints_per_part=joints_per_part,
                                       temporal_kernel=3)

        # Stage 2: Cross-part 1x1 mix
        self.cross_part = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 3: S-T Decoupled blocks
        self.st_block1 = STDecoupledBlock(64, 128)
        self.st_block2 = STDecoupledBlock(128, 256)
        self.st_block3 = STDecoupledBlock(256, 256)

        # Stage 4: Multi-scale temporal
        self.ms_temporal = MultiScaleTemporal(256, 256)

        # Stage 5: SE attention
        self.se = SqueezeExcitation(256)

        # Stage 6: Pool + heads
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.heads = nn.ModuleDict({
            name: nn.Linear(256, nc) for name, nc in heads.items()
        })

    def forward(self, x):
        # x: [B, 7, 60, 25]
        x = self.body_part(x)            # [B, 64, 60, 5]
        x = self.cross_part(x)            # [B, 64, 60, 5]

        x = self.st_block1(x)             # [B, 128, 60, 5]
        x = F.avg_pool2d(x, kernel_size=(2, 1))  # [B, 128, 30, 5]

        x = self.st_block2(x)             # [B, 256, 30, 5]
        x = F.avg_pool2d(x, kernel_size=(2, 1))  # [B, 256, 15, 5]

        x = self.st_block3(x)             # [B, 256, 15, 5]
        x = self.ms_temporal(x)           # [B, 256, 15, 5]
        x = self.se(x)                    # [B, 256, 15, 5]

        x = self.pool(x).flatten(1)       # [B, 256]
        return {name: head(x) for name, head in self.heads.items()}


def build_psp_net(heads, in_channels=7, num_parts=5, joints_per_part=5):
    return PSPNet(heads=heads, in_channels=in_channels,
                  num_parts=num_parts, joints_per_part=joints_per_part)


if __name__ == '__main__':
    # 빠른 sanity check
    heads = {'action_upper': 6, 'action_lower': 10, 'pose': 9, 'hand': 3, 'foot': 3}
    model = build_psp_net(heads)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PSP-Net 파라미터: {n_params/1e6:.2f}M")
    x = torch.randn(2, 7, 60, 25)
    out = model(x)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
