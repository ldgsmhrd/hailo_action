"""PSP-Net 학습 entry point (NTU60 / NTU120 통합).

사용:
    python scripts/train.py --config configs/ntu60_psp_mb4.yaml
    python scripts/train.py --config configs/ntu60_psp_mb4.yaml --seed 7
    python scripts/train.py --config configs/ntu60_psp_mb4.yaml --benchmark xview
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# 본 repository 를 PYTHONPATH 에 추가
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from psp_net.models.psp_net import build_psp_net_ntu
from psp_net.models.psp_mb import build_psp_mb
from psp_net.models.psp_mb4 import build_psp_mb4


def get_dataset_module(dataset_name):
    if dataset_name == 'ntu60':
        from psp_net.dataset.ntu_dataset import NTUDataset, make_split
    elif dataset_name == 'ntu120':
        from psp_net.dataset.ntu120_dataset import NTUDataset, make_split
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return NTUDataset, make_split


def build_model(model_name, num_classes, in_channels, base_ch):
    if model_name == 'psp_net':
        return build_psp_net_ntu(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'psp_mb_3d':
        return build_psp_mb(num_classes=num_classes, in_channels=in_channels,
                            base_ch=base_ch)
    elif model_name in ('psp_mb4', 'mb4'):
        return build_psp_mb4(num_classes=num_classes, in_channels=in_channels,
                             base_ch=base_ch)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='YAML config path')
    ap.add_argument('--seed', type=int, default=None, help='Override config seed')
    ap.add_argument('--benchmark', default=None, help='Override config benchmark (xsub/xview/cset)')
    ap.add_argument('--ckpt-dir', default='checkpoints', help='Where to save best ckpt')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = args.seed if args.seed is not None else cfg.get('seed', 42)
    benchmark = args.benchmark if args.benchmark is not None else cfg['data']['benchmark']

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    cfg_m, cfg_d, cfg_tr, cfg_aug = cfg['model'], cfg['data'], cfg['training'], cfg['augment']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    NTUDataset, make_split = get_dataset_module(cfg_d['dataset'])
    npy_root = cfg_d['npy_root']
    train_files, test_files = make_split(npy_root, benchmark=benchmark)
    print(f"dataset={cfg_d['dataset']} benchmark={benchmark}")
    print(f"  train: {len(train_files)}  test: {len(test_files)}")

    train_ds = NTUDataset(
        train_files, T=cfg_d['frames_per_clip'], augment=True,
        use_body=cfg_d['use_body'], use_3d=cfg_d['use_3d'],
        use_bone_motion=cfg_d.get('use_bone_motion', False),
        augment_cfg=cfg_aug,
    )
    test_ds = NTUDataset(
        test_files, T=cfg_d['frames_per_clip'], augment=False,
        use_body=cfg_d['use_body'], use_3d=cfg_d['use_3d'],
        use_bone_motion=cfg_d.get('use_bone_motion', False),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg_tr['batch_size'], shuffle=True,
                              num_workers=cfg_tr['num_workers'], drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg_tr['batch_size'], shuffle=False,
                             num_workers=cfg_tr['num_workers'])

    # Model
    model = build_model(cfg_m['name'], cfg_m['num_classes'],
                        cfg_m['in_channels'], cfg_m.get('base_ch', 64)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={cfg_m['name']}  params={n_params/1e6:.2f}M  seed={seed}")

    # Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg_tr.get('label_smoothing', 0.0))
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg_tr['lr'],
        momentum=cfg_tr['momentum'], nesterov=cfg_tr.get('nesterov', True),
        weight_decay=cfg_tr['weight_decay'],
    )
    warmup_epochs = cfg_tr.get('warmup_epochs', 5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg_tr['epochs'] - warmup_epochs,
    )
    mixup_alpha = cfg_tr.get('mixup_alpha', 0)
    mixup_prob = cfg_tr.get('mixup_prob', 0)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    tag = f"{cfg_d['dataset']}_{cfg_m['name']}_{benchmark}_s{seed}"
    save_path = os.path.join(args.ckpt_dir, f"best_{tag}.pth")

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(cfg_tr['epochs']):
        model.train()
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = cfg_tr['lr'] * (epoch + 1) / warmup_epochs
        loss_sum = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if mixup_alpha > 0 and np.random.rand() < mixup_prob:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[idx]
                y_a, y_b = y, y[idx]
                logits = model(x)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                pred = logits.argmax(1)
                correct += (lam * (pred == y_a).sum().item() +
                            (1 - lam) * (pred == y_b).sum().item())
            else:
                logits = model(x)
                loss = criterion(logits, y)
                correct += (logits.argmax(1) == y).sum().item()
            loss_sum += loss.item() * y.size(0)
            total += y.size(0)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch >= warmup_epochs:
            cosine.step()

        # Eval
        model.eval()
        t_correct = t_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                t_correct += (model(x).argmax(1) == y).sum().item()
                t_total += y.size(0)
        test_acc = 100 * t_correct / t_total
        print(f"epoch {epoch:>3}  loss={loss_sum/total:.3f}  "
              f"train={100*correct/total:.2f}%  test={test_acc:.2f}%  "
              f"lr={optimizer.param_groups[0]['lr']:.4f}", flush=True)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'test_acc': test_acc, 'config': cfg},
                       save_path)
            print(f"  * saved best={test_acc:.2f}%", flush=True)

    print(f"\nDONE: best={best_acc:.2f}% saved to {save_path}  "
          f"({(time.time()-t0)/60:.1f}분)")


if __name__ == '__main__':
    main()
