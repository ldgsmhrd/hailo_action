"""Multi-task test set 평가 v3 — 5 head accuracy + 클래스별 recall."""
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/ubuntu/safemotion/ResNet-ActionRecognition')

from configs.aica_multitask_config import (
    CONFIG, CATEGORIES,
    ACTION_UPPER_CLASSES, ACTION_LOWER_CLASSES,
    POSE_CLASSES, HAND_CLASSES, FOOT_CLASSES,
)
from scripts.dataset_mt import MultiTaskActionDataset
from scripts.model_mt import build_multitask_model
from dataset.encoders import PseudoImageEncoder


CLASS_NAMES = {
    'action_upper': ACTION_UPPER_CLASSES,
    'action_lower': ACTION_LOWER_CLASSES,
    'pose': POSE_CLASSES,
    'hand': HAND_CLASSES,
    'foot': FOOT_CLASSES,
}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = PseudoImageEncoder(order='tssi', channels=('pos', 'velocity', 'angle'))

    test_ds = MultiTaskActionDataset(
        split_root=CONFIG['paths']['split_root'], split='test',
        T=60, encoder=encoder, augment=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    model = build_multitask_model(heads=CATEGORIES, in_channels=encoder.num_channels,
                                  cfg=CONFIG['model']).to(device)
    ck = torch.load(os.path.join(CONFIG['paths']['model_dir'], 'best_mt.pth'),
                    map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    print(f"Loaded ckpt: epoch {ck.get('epoch')}  val_avg {ck.get('best_val_avg', 0):.4f}")
    print(f"Test set: {len(test_ds)} clips\n")

    all_preds = {k: [] for k in CATEGORIES}
    all_gts = {k: [] for k in CATEGORIES}
    with torch.no_grad():
        for x, labels in test_loader:
            x = x.to(device)
            out = model(x)
            for k in CATEGORIES:
                all_preds[k].extend(out[k].argmax(1).cpu().numpy().tolist())
                all_gts[k].extend(labels[k].numpy().tolist())

    overall_acc = {}
    print("=" * 70)
    for head in CATEGORIES:
        preds = np.array(all_preds[head])
        gts = np.array(all_gts[head])
        acc = (preds == gts).mean()
        overall_acc[head] = acc
        nc = CATEGORIES[head]
        names = CLASS_NAMES[head]
        print(f"\n=== {head} : test acc = {acc:.4f} ===")
        for c in range(nc):
            mask = gts == c
            if mask.sum() == 0:
                print(f"  cls {c} ({names[c]:<18}): n=0")
                continue
            cls_acc = (preds[mask] == c).mean()
            print(f"  cls {c} ({names[c]:<18}): n={int(mask.sum()):4d}  recall={cls_acc:.3f}")
    print("\n" + "=" * 70)
    print(f"평균 test acc = {np.mean(list(overall_acc.values())):.4f}")
    for k, v in overall_acc.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
