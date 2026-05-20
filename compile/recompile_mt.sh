#!/bin/bash
# Multi-task ResNet18 (5 head) compile.
# 입력  : action_resnet_mt.onnx (5 outputs: action_upper/lower/pose/hand/foot)
# 출력  : action_resnet_mt.hef
#
# 사전조건:
#   - Hailo Dataflow Compiler 가 설치된 docker 안에서 실행
#   - calibration_set.npy 가 (N, 60, 25, 7) shape 으로 존재

set -e
cd "$(dirname "$0")"

ONNX="action_resnet_mt.onnx"
CALIB="calibration_set.npy"
HW_ARCH="hailo8"

# 산출물 정리
for f in action_resnet_mt.har action_resnet_mt_quantized.har \
         action_resnet_mt.hef action_resnet_mt_compiled.har; do
    rm -f "$f"
done

echo "=== 1/3. Parse — ONNX → HAR ==="
hailo parser onnx "$ONNX" \
    --hw-arch "$HW_ARCH" \
    --har-path action_resnet_mt.har \
    --tensor-shapes "input=[1,7,60,25]"

echo ""
echo "=== 2/3. Optimize — 1500 calibration ==="
hailo optimize action_resnet_mt.har \
    --calib-set-path "$CALIB" \
    --output-har-path action_resnet_mt_quantized.har \
    --hw-arch "$HW_ARCH"

echo ""
echo "=== 3/3. Compile — HAR → HEF ==="
hailo compiler action_resnet_mt_quantized.har \
    --hw-arch "$HW_ARCH" \
    --output-dir .

if [ -f action_resnet_mt_quantized.hef ]; then
    mv action_resnet_mt_quantized.hef action_resnet_mt.hef
fi

echo ""
echo "=== 완료 ==="
ls -lh action_resnet_mt.hef
