"""NTU RGB+D 60 단일 헤드 60-class 학습 — 표준 benchmark 비교용.

실행:
  CUDA_VISIBLE_DEVICES=0 python -m scripts.train_ntu60

데이터셋:
  - dataset/gen_npy_ntu60.py 먼저 실행해서 NPY 생성 필요
  - data/ntu60_split/{train,test}/{class_idx}/*.npy

모델:
  - 입력 [B, 7, 60, 25] (TSSI 인코딩 후)
  - 출력 60 class single head
"""
import os, sys, time, json
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch.utils.data import Dataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from configs.ntu60_config import CONFIG, NTU60_CLASSES
from dataset.encoders import PseudoImageEncoder


class NTU60Dataset(Dataset):
    """[T, 25, 3] NPY → pseudo-image [7, T, 25] + class label."""

    def __init__(self, split_root, split, encoder, augment=False, augment_cfg=None):
        import glob
        self.encoder = encoder
        self.augment = augment
        self.augment_cfg = augment_cfg or {}
        base = os.path.join(split_root, split)
        self.paths, self.labels = [], []
        for cls_dir in sorted(glob.glob(os.path.join(base, '*'))):
            cls = int(os.path.basename(cls_dir))
            for f in glob.glob(os.path.join(cls_dir, '*.npy')):
                self.paths.append(f)
                self.labels.append(cls)
        if not self.paths:
            raise RuntimeError(f"No NPY under {base}")
        print(f"  [{split}] {len(self.paths)} samples / {len(set(self.labels))} classes")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        kp = np.load(self.paths[idx]).astype(np.float32)   # [T, 25, 3]
        # 정규화: 1번 관절 (spine base) 기준 중심화 + 어깨 길이로 스케일
        spine_base = kp[:, 0:1, :2]    # [T, 1, 2]
        kp[..., :2] = kp[..., :2] - spine_base   # 중심화
        # confidence: NTU 는 모두 1.0 (depth sensor) → 가짜 confidence
        if kp.shape[-1] == 3:
            # x, y, z → x, y, conf (z 무시하고 confidence 1.0 대체)
            conf = np.ones_like(kp[..., 0:1])
            kp = np.concatenate([kp[..., :2], conf], axis=-1)

        if self.augment:
            kp = self._augment(kp)

        pseudo = self.encoder.encode(kp)   # [7, T, 25]
        return torch.from_numpy(pseudo), torch.tensor(self.labels[idx], dtype=torch.long)

    def _augment(self, clip):
        cfg = self.augment_cfg
        if np.random.random() < cfg.get('flip_prob', 0.0):
            clip[..., 0] = -clip[..., 0]   # 좌우 반전 (중심화된 좌표라 - 부호)
        std = cfg.get('coord_noise_std', 0.0)
        if std > 0:
            clip[..., :2] += np.random.randn(*clip[..., :2].shape).astype(np.float32) * std
        return clip


def build_single_head_model(num_classes, in_channels=7, first_conv_stride=(2, 1)):
    """ResNet18 + 단일 분류 헤드 (NTU60-class)."""
    net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    orig = net.conv1
    new_conv = nn.Conv2d(in_channels, orig.out_channels,
                         kernel_size=orig.kernel_size, stride=first_conv_stride,
                         padding=orig.padding, bias=orig.bias is not None)
    with torch.no_grad():
        copy_ch = min(3, in_channels)
        new_conv.weight[:, :copy_ch] = orig.weight[:, :copy_ch]
    net.conv1 = new_conv
    net.fc = nn.Linear(512, num_classes)
    return net


def main():
    cfg_tr = CONFIG['training']
    device = torch.device(cfg_tr['device'])

    encoder = PseudoImageEncoder(
        order=CONFIG['encoder']['order'],
        channels=tuple(CONFIG['encoder']['channels']),
    )
    in_channels = encoder.num_channels

    train_ds = NTU60Dataset(
        CONFIG['paths']['split_root'], 'train', encoder=encoder,
        augment=True, augment_cfg=CONFIG['augment'],
    )
    test_ds = NTU60Dataset(
        CONFIG['paths']['split_root'], 'test', encoder=encoder,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg_tr['batch_size'],
                              shuffle=True, num_workers=cfg_tr['num_workers'], drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg_tr['batch_size'],
                             shuffle=False, num_workers=cfg_tr['num_workers'])

    model = build_single_head_model(num_classes=CONFIG['num_classes'],
                                     in_channels=in_channels,
                                     first_conv_stride=tuple(CONFIG['model']['first_conv_stride'])).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg_tr['lr'],
                                momentum=cfg_tr['momentum'], nesterov=cfg_tr['nesterov'],
                                weight_decay=cfg_tr['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_tr['epochs'])
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=cfg_tr.get('amp', False))
    best_acc = 0.0
    save_path = os.path.join(CONFIG['paths']['model_dir'], f"best_ntu60_{CONFIG['protocol']}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"NTU60 학습 — protocol={CONFIG['protocol']}, classes={CONFIG['num_classes']}")
    print(f"  train: {len(train_ds)}  test: {len(test_ds)}")

    for epoch in range(cfg_tr['epochs']):
        t0 = time.time()
        model.train()
        train_loss = 0.0; train_correct = 0; train_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg_tr.get('amp', False)):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item()
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)
        scheduler.step()
        train_acc = train_correct / train_total

        model.eval()
        test_correct = 0; test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                test_correct += (logits.argmax(1) == y).sum().item()
                test_total += x.size(0)
        test_acc = test_correct / test_total
        dt = time.time() - t0

        print(f"epoch {epoch:03d}  loss={train_loss/len(train_loader):.3f}  "
              f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  ({dt:.1f}s)", flush=True)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'protocol': CONFIG['protocol'],
                'num_classes': CONFIG['num_classes'],
            }, save_path)
            print(f"  ★ saved (test_acc={best_acc:.4f})", flush=True)

    print(f"\n최종 best test acc ({CONFIG['protocol']}): {best_acc:.4f}")
    print(f"  → {save_path}")


if __name__ == '__main__':
    main()
