"""HEF (INT8) test set 평가 — PyTorch 모델과 직접 비교.

실행 (보드에서):
  python3 /app/npu/hef_test_eval.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, "/app/npu")

from hailo_platform import Device, InferVStreams
from init_hailo import init_action_npu


MT_CLASSES_LEN = {
    'action_upper': 6,
    'action_lower': 10,
    'pose': 9,
    'hand': 3,
    'foot': 3,
}

# Hailo 컴파일러의 fcN 부여 순서는 ModuleDict 순서와 무관 — shape 기반 자동 매핑
# 실제 HEF 출력 shape:
#   fc1=(10,) action_lower / fc2=(6,) action_upper / fc3=(3,) foot
#   fc4=(3,) hand / fc5=(9,) pose
# hand/foot 같은 3-class 클래스 둘 사이 구분은 일단 보드 단일 검증으로 확정
FC_TO_HEAD = {
    'fc1': 'action_lower',
    'fc2': 'action_upper',
    'fc3': 'foot',
    'fc4': 'hand',
    'fc5': 'pose',
}


def main():
    npz_path = '/share/test_batch.npz'
    hef_path = '/app/models/action_resnet_mt.hef'

    data = np.load(npz_path)
    xs = data['x']                        # [N, 7, 60, 25]
    labels = {k: data[k] for k in MT_CLASSES_LEN}
    N = xs.shape[0]
    print(f"Loaded {N} test samples from {npz_path}")
    print(f"  x shape: {xs.shape}, dtype: {xs.dtype}")

    # HEF 는 NHWC 입력 기대 (Hailo 변환에서 transpose) → [N, 60, 25, 7]
    xs_nhwc = np.transpose(xs, (0, 2, 3, 1)).astype(np.float32)
    print(f"  NHWC shape: {xs_nhwc.shape}")

    devs = Device.scan()
    if not devs:
        print("ERROR: no NPU"); return
    action_h = init_action_npu(device_id=devs[1] if len(devs) > 1 else devs[0],
                               npu_index=1, action_hef_path=hef_path)

    # head → vstream name 매핑
    head_to_vsname = {}
    for o in action_h['output_infos']:
        suffix = o.name.rsplit('/', 1)[-1]
        head = FC_TO_HEAD.get(suffix)
        if head:
            head_to_vsname[head] = o.name
    print(f"head→vstream: {head_to_vsname}")

    in_name = action_h['input_info'].name

    preds = {k: np.empty(N, dtype=np.int64) for k in MT_CLASSES_LEN}

    t0 = time.time()
    with action_h['network_group'].activate(action_h['network_group_params']), \
         InferVStreams(action_h['network_group'],
                       action_h['input_params'],
                       action_h['output_params']) as pipe:
        for i in range(N):
            if i % 200 == 0:
                el = time.time() - t0
                rate = i / max(el, 1e-6)
                print(f"  {i}/{N}  ({rate:.1f} it/s)", flush=True)
            inp = np.ascontiguousarray(xs_nhwc[i:i+1])   # [1, 60, 25, 7]
            out = pipe.infer({in_name: inp})
            for head, vs in head_to_vsname.items():
                logits = np.asarray(out[vs]).reshape(-1)
                preds[head][i] = int(logits.argmax())

    el = time.time() - t0
    print(f"\nInference done in {el:.1f}s  ({N/el:.1f} samples/s)\n")

    # 결과
    print("=" * 60)
    print(f"HEF (INT8 quantized) test results — N={N}")
    print("=" * 60)
    overall = {}
    for head in MT_CLASSES_LEN:
        acc = (preds[head] == labels[head]).mean()
        overall[head] = acc
        nc = MT_CLASSES_LEN[head]
        per_class = []
        for c in range(nc):
            mask = labels[head] == c
            if mask.sum() > 0:
                cls_acc = (preds[head][mask] == c).mean()
                per_class.append(f"{c}:{cls_acc:.2f}({int(mask.sum())})")
            else:
                per_class.append(f"{c}:n=0")
        print(f"  {head:<14}: {acc*100:.2f}%  per-class: {' '.join(per_class)}")
    print("=" * 60)
    print(f"  평균          : {100*np.mean(list(overall.values())):.2f}%")

    # PyTorch 비교
    PYTORCH = {
        'action_upper': 0.9728,
        'action_lower': 0.9543,
        'pose':         0.8673,
        'hand':         0.9985,
        'foot':         0.9961,
    }
    print(f"\n=== quantization gap (HEF - PyTorch) ===")
    for head in MT_CLASSES_LEN:
        gap = (overall[head] - PYTORCH[head]) * 100
        print(f"  {head:<14}: PyTorch {PYTORCH[head]*100:.2f}% → HEF {overall[head]*100:.2f}% (gap {gap:+.2f}%)")


if __name__ == '__main__':
    main()
