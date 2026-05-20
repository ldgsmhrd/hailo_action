"""Multi-task ResNet18 — 단일 backbone + 여러 head."""
import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskActionResNet(nn.Module):
    """
    입력 [B, C, T, J] → backbone (ResNet18) → feature [B, 512]
       → 각 head: Linear(512, num_classes_per_head)
       → dict { head_name: [B, num_classes] }
    """
    def __init__(self, heads, in_channels=7, backbone='resnet18',
                 pretrained=True, first_conv_stride=(2, 1)):
        super().__init__()
        self.heads_info = heads   # dict {name: num_classes}

        if backbone == 'resnet18':
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        else:
            raise ValueError(f"backbone {backbone} not supported in multi-task")

        # 첫 conv 채널 변경
        orig = net.conv1
        new_conv = nn.Conv2d(in_channels, orig.out_channels,
                             kernel_size=orig.kernel_size, stride=first_conv_stride,
                             padding=orig.padding, bias=orig.bias is not None)
        if pretrained:
            with torch.no_grad():
                copy_ch = min(3, in_channels)
                new_conv.weight[:, :copy_ch] = orig.weight[:, :copy_ch]
        net.conv1 = new_conv
        net.fc = nn.Identity()   # backbone 만 사용
        self.backbone = net

        # 5 head
        self.heads = nn.ModuleDict({
            name: nn.Linear(feat_dim, num_classes)
            for name, num_classes in heads.items()
        })

    def forward(self, x):
        feat = self.backbone(x)   # [B, 512]
        return {name: head(feat) for name, head in self.heads.items()}


def build_multitask_model(heads, in_channels=7, cfg=None):
    cfg = cfg or {}
    return MultiTaskActionResNet(
        heads=heads,
        in_channels=in_channels,
        backbone=cfg.get('backbone', 'resnet18'),
        pretrained=cfg.get('pretrained', True),
        first_conv_stride=tuple(cfg.get('first_conv_stride', (2, 1))),
    )
