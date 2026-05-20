"""Multi-task test 분석 — 각 클래스별 데이터 수 + 오분류 top-3."""
import os, sys, numpy as np
import torch
from collections import Counter
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


KR = {
    'action_upper': {0:'없음',1:'펀치',2:'손흔들기',3:'손뼉치기',4:'손올리기',5:'손내리기'},
    'action_lower': {0:'없음',1:'서성이기',2:'걷기',3:'달리기',4:'점프-제자리',
                     5:'넘어짐',6:'킥',7:'점프-두발',8:'외발점프',9:'외발점프-제자리'},
    'pose':         {0:'바닥앉기',1:'의자앉기',2:'무릎꿇기',3:'무릎서기',4:'서있기',
                     5:'허리구부리기',6:'누워있기',7:'무릎기기',8:'기타'},
    'hand':         {0:'없음',1:'팔짱끼기',2:'양팔들기'},
    'foot':         {0:'없음',1:'다리꼬기',2:'한쪽다리들기'},
}
HEAD_KR = {'action_upper':'상체','action_lower':'하체','pose':'자세','hand':'손','foot':'발'}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = PseudoImageEncoder(order='tssi', channels=('pos','velocity','angle'))

    # train + val + test 카운트 따로
    counts = {sp: {h: Counter() for h in CATEGORIES} for sp in ('train','val','test')}
    for sp in counts:
        ds = MultiTaskActionDataset(
            split_root=CONFIG['paths']['split_root'], split=sp,
            T=60, encoder=encoder, augment=False)
        for lbl in ds.labels:
            for h in CATEGORIES:
                counts[sp][h][lbl[h]] += 1

    # test 만 추론으로 confusion 만들기
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

    preds = {k: [] for k in CATEGORIES}
    gts = {k: [] for k in CATEGORIES}
    with torch.no_grad():
        for x, labels in test_loader:
            x = x.to(device)
            out = model(x)
            for k in CATEGORIES:
                preds[k].extend(out[k].argmax(1).cpu().numpy().tolist())
                gts[k].extend(labels[k].numpy().tolist())

    # 헤드별 confusion + 데이터 수 표
    for head in CATEGORIES:
        nc = CATEGORIES[head]
        p = np.array(preds[head]); g = np.array(gts[head])
        cm = np.zeros((nc, nc), dtype=np.int64)
        for i in range(len(g)):
            cm[g[i], p[i]] += 1

        print(f"\n{'='*70}")
        print(f"[{HEAD_KR[head]}] {head}  —  {nc} 클래스")
        print(f"{'='*70}")
        # 헤더
        cnt_tr = counts['train'][head]; cnt_va = counts['val'][head]; cnt_te = counts['test'][head]
        print(f"{'idx':>3}  {'한글(영어)':<26}  {'train':>6} {'val':>5} {'test':>5}  {'recall':>7}  헷갈린 top-3")
        for c in range(nc):
            kn = KR[head].get(c, f'cls{c}')
            n_te = int(g.tolist().count(c))
            tot_pred = cm[c].sum()
            recall = cm[c,c] / max(tot_pred, 1) if tot_pred else 0.0
            # 오분류 top-3 (자기 제외)
            wrong = [(j, cm[c,j]) for j in range(nc) if j != c and cm[c,j] > 0]
            wrong.sort(key=lambda t: -t[1])
            top3 = ', '.join(f"{KR[head].get(j,j)}({n})" for j,n in wrong[:3]) if wrong else '-'
            print(f"  {c:>1}  {kn:<26}  {cnt_tr.get(c,0):>6} {cnt_va.get(c,0):>5} {n_te:>5}  {recall*100:>6.1f}%  {top3}")


if __name__ == '__main__':
    main()
