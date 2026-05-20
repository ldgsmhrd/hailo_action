"""PSP-Net 학습 entry — 자체 데이터셋 (5 head multi-task).

실행:
  CUDA_VISIBLE_DEVICES=0 python -m psp_net.scripts.train_psp
"""
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _REPO)

from psp_net.configs.psp_config import CONFIG, CATEGORIES
from psp_net.dataset.psp_dataset import PSPDataset
from psp_net.models.psp_net import build_psp_net


def compute_class_weights(dataset, head, num_classes, beta=0.999):
    cnt = Counter()
    for lbl in dataset.labels:
        cnt[lbl[head]] += 1
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        n = cnt.get(c, 0)
        if n == 0:
            weights[c] = 0.0
        else:
            eff_num = (1.0 - beta ** n) / (1.0 - beta)
            weights[c] = 1.0 / eff_num
    nonzero = weights[weights > 0]
    if len(nonzero) > 0:
        weights = weights * (len(nonzero) / nonzero.sum())
    return torch.from_numpy(weights)


def main():
    cfg_tr = CONFIG['training']
    device = torch.device(cfg_tr['device'])

    train_ds = PSPDataset(
        split_root=CONFIG['paths']['split_root'],
        split='train',
        T=CONFIG['frames_per_clip'],
        augment=True,
        augment_cfg=CONFIG['augment'],
    )
    val_ds = PSPDataset(
        split_root=CONFIG['paths']['split_root'],
        split='val',
        T=CONFIG['frames_per_clip'],
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg_tr['batch_size'],
                              shuffle=True, num_workers=cfg_tr['num_workers'],
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg_tr['batch_size'],
                            shuffle=False, num_workers=cfg_tr['num_workers'])

    model = build_psp_net(
        heads=CATEGORIES,
        in_channels=CONFIG['in_channels'],
        num_parts=CONFIG['num_parts'],
        joints_per_part=CONFIG['joints_per_part'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 70)
    print(f"PSP-Net")
    print(f"  파라미터 : {n_params/1e6:.2f}M  (baseline ResNet18 11.2M 대비 1/{11.2/(n_params/1e6):.1f})")
    print(f"  heads    : {CATEGORIES}")
    print(f"  loss w   : {cfg_tr['loss_weights']}")
    print(f"  train    : {len(train_ds)}  val: {len(val_ds)}")
    print("=" * 70)

    # per-head class weight
    criterions = {}
    use_cw = cfg_tr.get('use_class_weights', False)
    for head, nc in CATEGORIES.items():
        if use_cw:
            cw = compute_class_weights(train_ds, head, nc).to(device)
            criterions[head] = nn.CrossEntropyLoss(weight=cw)
        else:
            criterions[head] = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg_tr['lr'],
        momentum=cfg_tr['momentum'],
        nesterov=cfg_tr['nesterov'],
        weight_decay=cfg_tr['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_tr['epochs'])

    loss_weights = cfg_tr['loss_weights']
    best_avg = 0.0
    os.makedirs(CONFIG['paths']['model_dir'], exist_ok=True)
    save_path = os.path.join(CONFIG['paths']['model_dir'], 'best_psp.pth')

    for epoch in range(cfg_tr['epochs']):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_correct = {k: 0 for k in CATEGORIES}
        train_total = 0
        for x, labels in train_loader:
            x = x.to(device, non_blocking=True)
            labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
            out = model(x)
            losses = {h: criterions[h](out[h], labels[h]) for h in CATEGORIES}
            total_loss = sum(loss_weights.get(k, 1.0) * v for k, v in losses.items())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss_sum += total_loss.item()
            for h in CATEGORIES:
                pred = out[h].argmax(1)
                train_correct[h] += (pred == labels[h]).sum().item()
            train_total += x.size(0)

        model.eval()
        val_correct = {k: 0 for k in CATEGORIES}
        val_total = 0
        with torch.no_grad():
            for x, labels in val_loader:
                x = x.to(device, non_blocking=True)
                labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
                out = model(x)
                for h in CATEGORIES:
                    pred = out[h].argmax(1)
                    val_correct[h] += (pred == labels[h]).sum().item()
                val_total += x.size(0)

        scheduler.step()
        dt = time.time() - t0
        train_accs = {k: v / train_total for k, v in train_correct.items()}
        val_accs = {k: v / val_total for k, v in val_correct.items()}
        val_avg = sum(val_accs.values()) / len(val_accs)

        log = (f"epoch {epoch:03d}  loss={train_loss_sum/len(train_loader):.3f}  "
               f"val[avg={val_avg:.3f} lower={val_accs['action_lower']:.3f} "
               f"upper={val_accs['action_upper']:.3f} pose={val_accs['pose']:.3f} "
               f"hand={val_accs['hand']:.3f} foot={val_accs['foot']:.3f}]  ({dt:.1f}s)")
        print(log, flush=True)

        if val_avg > best_avg:
            best_avg = val_avg
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'val_accs': val_accs,
                'best_val_avg': best_avg,
                'heads': CATEGORIES,
                'config': CONFIG,
            }, save_path)
            print(f"  ★ saved  val_avg={best_avg:.4f}", flush=True)

    print(f"\n학습 완료. best val avg = {best_avg:.4f}")
    print(f"  → {save_path}")


if __name__ == '__main__':
    main()
