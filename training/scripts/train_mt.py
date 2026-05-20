"""Multi-task 학습 entry v3 — 5 head + per-head class weight (imbalance 완화).
실행: CUDA_VISIBLE_DEVICES=3 python -m scripts.train_mt
"""
import os, sys, json, time
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from configs.aica_multitask_config import CONFIG, CATEGORIES
from scripts.dataset_mt import MultiTaskActionDataset
from scripts.model_mt import build_multitask_model
from dataset.encoders import PseudoImageEncoder


def compute_class_weights(dataset, head, num_classes, beta=0.999):
    """effective number of samples 기반 class weight (Cui et al. 2019)."""
    cnt = Counter()
    for lbl in dataset.labels:
        cnt[lbl[head]] += 1
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        n = cnt.get(c, 0)
        if n == 0:
            weights[c] = 0.0   # 빈 class 는 loss 기여 0
        else:
            eff_num = (1.0 - beta ** n) / (1.0 - beta)
            weights[c] = 1.0 / eff_num
    # 평균 가중치를 1.0 으로 맞춤 (zero 제외)
    nonzero = weights[weights > 0]
    if len(nonzero) > 0:
        weights = weights * (len(nonzero) / nonzero.sum())
    return torch.from_numpy(weights)


def main():
    cfg_tr = CONFIG['training']
    device = torch.device(cfg_tr['device'])

    encoder = PseudoImageEncoder(
        order=CONFIG['encoder']['order'],
        channels=tuple(CONFIG['encoder']['channels']),
    )
    in_channels = encoder.num_channels

    train_ds = MultiTaskActionDataset(
        split_root=CONFIG['paths']['split_root'], split='train',
        T=CONFIG['frames_per_clip'], encoder=encoder,
        augment=True, augment_cfg=CONFIG['augment'],
    )
    val_ds = MultiTaskActionDataset(
        split_root=CONFIG['paths']['split_root'], split='val',
        T=CONFIG['frames_per_clip'], encoder=encoder, augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg_tr['batch_size'], shuffle=True,
                              num_workers=cfg_tr['num_workers'], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg_tr['batch_size'], shuffle=False,
                            num_workers=cfg_tr['num_workers'])

    model = build_multitask_model(
        heads=CATEGORIES, in_channels=in_channels,
        cfg=CONFIG['model']
    ).to(device)

    # ---- per-head class weight (effective-number) ----
    criterions = {}
    use_cw = cfg_tr.get('use_class_weights', False)
    print("=" * 60)
    print(f"Multi-task ResNet18 v3")
    print(f"  heads: {CATEGORIES}")
    print(f"  loss weights: {cfg_tr['loss_weights']}")
    print(f"  class weights: {use_cw}")
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")
    for head, nc in CATEGORIES.items():
        if use_cw:
            cw = compute_class_weights(train_ds, head, nc).to(device)
            criterions[head] = nn.CrossEntropyLoss(weight=cw)
            cnt = Counter([l[head] for l in train_ds.labels])
            print(f"  [{head}] cnt={[cnt.get(c,0) for c in range(nc)]}  cw={cw.cpu().numpy().round(2).tolist()}")
        else:
            criterions[head] = nn.CrossEntropyLoss()
    print("=" * 60)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg_tr['lr'], momentum=cfg_tr['momentum'],
        nesterov=cfg_tr['nesterov'], weight_decay=cfg_tr['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_tr['epochs'])

    loss_weights = cfg_tr['loss_weights']
    best_val = 0.0
    save_path = os.path.join(CONFIG['paths']['model_dir'], 'best_mt.pth')

    for epoch in range(cfg_tr['epochs']):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_correct = {k: 0 for k in CATEGORIES}
        train_total = 0
        for x, labels in train_loader:
            x = x.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            out = model(x)
            losses = {head: criterions[head](out[head], labels[head]) for head in CATEGORIES}
            total_loss = sum(loss_weights.get(k, 1.0) * v for k, v in losses.items())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss_sum += total_loss.item()
            for head in CATEGORIES:
                pred = out[head].argmax(1)
                train_correct[head] += (pred == labels[head]).sum().item()
            train_total += x.size(0)

        model.eval()
        val_correct = {k: 0 for k in CATEGORIES}
        val_total = 0
        with torch.no_grad():
            for x, labels in val_loader:
                x = x.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}
                out = model(x)
                for head in CATEGORIES:
                    pred = out[head].argmax(1)
                    val_correct[head] += (pred == labels[head]).sum().item()
                val_total += x.size(0)

        scheduler.step()
        dt = time.time() - t0
        train_accs = {k: v/train_total for k, v in train_correct.items()}
        val_accs = {k: v/val_total for k, v in val_correct.items()}
        # 평균 val acc 를 best 기준 (action_lower 단독이 아니라 종합)
        val_main = sum(val_accs.values()) / len(val_accs)

        log = (f"epoch {epoch:03d}  loss={train_loss_sum/len(train_loader):.3f}  "
               f"val[avg={val_main:.3f} lower={val_accs['action_lower']:.3f} "
               f"upper={val_accs['action_upper']:.3f} pose={val_accs['pose']:.3f} "
               f"hand={val_accs['hand']:.3f} foot={val_accs['foot']:.3f}]  ({dt:.1f}s)")
        print(log, flush=True)

        if val_main > best_val:
            best_val = val_main
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'val_accs': val_accs,
                'best_val_avg': best_val,
                'encoder': {'order': CONFIG['encoder']['order'], 'channels': CONFIG['encoder']['channels']},
                'heads': CATEGORIES,
            }, save_path)
            print(f"  ★ saved (val avg={best_val:.4f})", flush=True)

    print(f"\n학습 완료. best val avg = {best_val:.4f}")
    print(f"  → {save_path}")


if __name__ == '__main__':
    main()
