"""PSP-Net Multi-Branch (단일 모델 2-stream).

NPU 호환 op 만 사용 (Conv2d, BN, ReLU, AdaptiveAvgPool, Sigmoid, Linear, Concat, Slice).

입력 채널 layout (use_3d=True + use_body=-1 + use_bone_motion=True):
  per body 12채널: [x,y,z, vx,vy,vz, bx,by,bz, bvx,bvy,bvz]
  2 body concat = 24 채널

  body 0: ch 0~11 = [pos 0:3, motion 3:6, bone 6:9, bone_motion 9:12]
  body 1: ch 12~23 = 동일

Stream 분리:
  Joint stream  = body0 [0:6] + body1 [12:18] = 12 ch (pos + motion)
  Bone stream   = body0 [6:12] + body1 [18:24] = 12 ch (bone + bone motion)

각 branch 가 mini PSP-Net 처리 → fusion → 단일 head.
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


class PSPNetMB(nn.Module):
    """Multi-Branch 단일 모델.

    입력: [B, 24, 64, 25]  (3D + 2body + bone_motion)
    출력: [B, num_classes]

    Branch 분리:
      Joint stream:  body0 [0:6] + body1 [12:18] = 12 ch
      Bone stream:   body0 [6:12] + body1 [18:24] = 12 ch
    """

    def __init__(self, num_classes=60, in_channels=24, base_ch=64,
                 joint_indices=None, bone_indices=None):
        super().__init__()
        # 채널 인덱스 (기본: 24ch full2+BM 가정)
        if joint_indices is None:
            joint_indices = list(range(0, 6)) + list(range(12, 18))
        if bone_indices is None:
            bone_indices = list(range(6, 12)) + list(range(18, 24))
        self.joint_indices = joint_indices
        self.bone_indices = bone_indices
        joint_ch = len(joint_indices)
        bone_ch = len(bone_indices)
        assert joint_ch + bone_ch <= in_channels, "stream index 합이 in_channels 초과"

        # 두 stream branch
        self.joint_branch = StreamBranch(joint_ch, base_ch)
        self.bone_branch = StreamBranch(bone_ch, base_ch)

        # Fusion
        fused_ch = self.joint_branch.out_ch + self.bone_branch.out_ch  # = base*4
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_ch, base_ch * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            STDecoupledBlock(base_ch * 4, base_ch * 4),
        )
        self.se = SqueezeExcitation(base_ch * 4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(base_ch * 4, num_classes)

        # buffer 로 인덱스 저장 (ONNX 호환)
        self.register_buffer('joint_idx_tensor',
                             torch.tensor(joint_indices, dtype=torch.long))
        self.register_buffer('bone_idx_tensor',
                             torch.tensor(bone_indices, dtype=torch.long))

    def forward(self, x):
        # x: [B, in_channels, T, J]
        # Stream split (index_select 는 ONNX 호환)
        x_joint = x.index_select(1, self.joint_idx_tensor)
        x_bone = x.index_select(1, self.bone_idx_tensor)

        # 2 branch parallel
        f_j = self.joint_branch(x_joint)    # [B, base*2, 16, 5]
        f_b = self.bone_branch(x_bone)      # [B, base*2, 16, 5]

        # Fusion
        f = torch.cat([f_j, f_b], dim=1)    # [B, base*4, 16, 5]
        f = self.fusion(f)
        f = self.se(f)

        f = self.pool(f).flatten(1)         # [B, base*4]
        f = self.dropout(f)
        return self.head(f)


def build_psp_mb(num_classes=60, in_channels=24, base_ch=64):
    return PSPNetMB(num_classes=num_classes, in_channels=in_channels, base_ch=base_ch)


if __name__ == '__main__':
    model = build_psp_mb(num_classes=60, in_channels=24, base_ch=64)
    n = sum(p.numel() for p in model.parameters())
    print(f"PSP-Net Multi-Branch 파라미터: {n/1e6:.2f}M")
    x = torch.randn(2, 24, 64, 25)
    out = model(x)
    print(f"  input  {tuple(x.shape)}")
    print(f"  output {tuple(out.shape)}")

    # ONNX 호환 sanity check
    try:
        torch.onnx.export(model, x[:1], '_test_psp_mb.onnx',
                          input_names=['input'], output_names=['logits'],
                          opset_version=11, dynamo=False, do_constant_folding=True)
        print("  ONNX export OK")
    except Exception as e:
        print(f"  ONNX export 실패: {e}")
