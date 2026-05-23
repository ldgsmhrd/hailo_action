"""PSP-Net 평가 entry point (PyTorch FP32).

사용:
    python scripts/eval.py --config configs/ntu60_psp_mb4.yaml --ckpt checkpoints/best.pth
    python scripts/eval.py --config configs/ntu60_psp_mb4.yaml --ckpt best.pth --tta
"""
import argparse
import os
import sys
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts.train import get_dataset_module, build_model


# NTU 좌/우 관절 인덱스 (raw, body-part reorder 전)
_L_IDX = np.array([4, 5, 6, 7, 21, 22, 12, 13, 14, 15])
_R_IDX = np.array([8, 9, 10, 11, 23, 24, 16, 17, 18, 19])


class TTAWrap(Dataset):
    """Anatomical horizontal mirror (x sign flip + L/R joint swap)."""
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def _flip_raw(self, sk):
        sk = sk.copy()
        sk[..., 0] = -sk[..., 0]
        sk[:, :, _L_IDX], sk[:, :, _R_IDX] = (
            sk[:, :, _R_IDX].copy(), sk[:, :, _L_IDX].copy())
        return sk
    def __getitem__(self, i):
        x_orig, y = self.base[i]
        path, _ = self.base.files[i]
        raw = np.load(path)[..., :3 if self.base.use_3d else 2]
        sk_flip = self._flip_raw(raw)
        if self.base.use_body == -1:
            x0 = self.base._make_psp_input(sk_flip[0])
            x1 = self.base._make_psp_input(sk_flip[1])
            xf = np.concatenate([x0, x1], 0)
        else:
            xf = self.base._make_psp_input(sk_flip[self.base.use_body])
        return x_orig, torch.from_numpy(xf), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--benchmark', default=None)
    ap.add_argument('--tta', action='store_true', help='Apply test-time augmentation')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg_m, cfg_d = cfg['model'], cfg['data']
    benchmark = args.benchmark if args.benchmark else cfg_d['benchmark']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    NTUDataset, make_split = get_dataset_module(cfg_d['dataset'])
    _, test_files = make_split(cfg_d['npy_root'], benchmark=benchmark)
    base = NTUDataset(
        test_files, T=cfg_d['frames_per_clip'], augment=False,
        use_body=cfg_d['use_body'], use_3d=cfg_d['use_3d'],
        use_bone_motion=cfg_d.get('use_bone_motion', False),
    )
    ds = TTAWrap(base) if args.tta else base
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8)

    model = build_model(cfg_m['name'], cfg_m['num_classes'],
                        cfg_m['in_channels'], cfg_m.get('base_ch', 64)).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck['model'] if isinstance(ck, dict) and 'model' in ck else ck)
    model.eval()
    print(f"Loaded: {args.ckpt}")
    print(f"  test: {len(base)}  TTA: {args.tta}")

    correct_no = correct_tta = total = 0
    with torch.no_grad():
        for batch in loader:
            if args.tta:
                x, xf, y = batch
                x, xf, y = x.to(device), xf.to(device), y.to(device)
                lo = model(x); lf = model(xf)
                lo_tta = (lo + lf) / 2.0
                correct_no += (lo.argmax(1) == y).sum().item()
                correct_tta += (lo_tta.argmax(1) == y).sum().item()
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                correct_no += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)

    print(f"\nResult ({benchmark}):")
    print(f"  No-TTA: {100*correct_no/total:.2f}%  ({correct_no}/{total})")
    if args.tta:
        print(f"  TTA:    {100*correct_tta/total:.2f}%  ({correct_tta}/{total})")
        print(f"  Gain:   +{100*(correct_tta-correct_no)/total:.2f}%p")


if __name__ == '__main__':
    main()
