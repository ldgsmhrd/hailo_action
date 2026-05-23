#!/bin/bash
# PSP-Net ONNX → Hailo HEF 컴파일
#
# Hailo Dataflow Compiler docker 컨테이너 안에서 실행:
#   docker run --rm -v $(pwd):/work hailo8_ai_sw_suite_2025-10:1 \
#       bash -c "cd /work && bash deploy/compile_hef.sh hailo8"
#
# 사용:
#   bash deploy/compile_hef.sh hailo8       # Hailo-8 용 (검증 보드)
#   bash deploy/compile_hef.sh hailo8l      # Hailo-8L 용 (Pi5)
#   bash deploy/compile_hef.sh hailo8 mb_3d # 특정 모델만 (default: mb4)

set -e
HW_ARCH="${1:-hailo8}"
MODEL="${2:-mb4}"
WORK_DIR="${WORK_DIR:-.}"
cd "$WORK_DIR"

if [ "$HW_ARCH" = "hailo8l" ]; then
    SUFFIX="_h8l"
else
    SUFFIX=""
fi

case "$MODEL" in
    mb4|psp_mb4)
        ONNX="psp_mb4.onnx"
        SHAPE="1,24,64,25"
        OUT="psp_mb4${SUFFIX}"
        ALLS="deploy/psp_mb_3d.alls"
        ;;
    mb_3d|psp_mb_3d)
        ONNX="psp_mb_3d.onnx"
        SHAPE="1,24,64,25"
        OUT="psp_mb_3d${SUFFIX}"
        ALLS="deploy/psp_mb_3d.alls"
        ;;
    mb_2d|psp_mb_2d)
        ONNX="psp_mb_2d.onnx"
        SHAPE="1,16,64,25"
        OUT="psp_mb_2d${SUFFIX}"
        ALLS="deploy/psp_mb_2d.alls"
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Supported: mb4 (default), mb_3d, mb_2d"
        exit 1
        ;;
esac

CALIB="${CALIB:-calib.npy}"

if [ ! -f "$ONNX" ]; then
    echo "ERROR: $ONNX not found. Run scripts/export_onnx.py first."
    exit 1
fi
if [ ! -f "$CALIB" ]; then
    echo "ERROR: $CALIB not found."
    echo "Generate calibration set: python scripts/build_calibration.py --npy-root data/ntu60/npy --out calib.npy"
    exit 1
fi

echo "================================================================"
echo "Compiling: $ONNX  ->  ${OUT}.hef  ($HW_ARCH)"
echo "================================================================"

rm -f ${OUT}.har ${OUT}_quantized.har ${OUT}.hef ${OUT}_compiled.har

# 1. Parse
echo "-- 1/3 parse --"
hailo parser onnx "$ONNX" --hw-arch "$HW_ARCH" \
    --har-path ${OUT}.har --tensor-shapes "input=[$SHAPE]"

# 2. Optimize
echo "-- 2/3 optimize --"
if [ -f "$ALLS" ]; then
    hailo optimize ${OUT}.har --calib-set-path "$CALIB" \
        --model-script "$ALLS" --output-har-path ${OUT}_quantized.har \
        --hw-arch "$HW_ARCH"
else
    hailo optimize ${OUT}.har --calib-set-path "$CALIB" \
        --output-har-path ${OUT}_quantized.har --hw-arch "$HW_ARCH"
fi

# 3. Compile
echo "-- 3/3 compile --"
hailo compiler ${OUT}_quantized.har --hw-arch "$HW_ARCH" --output-dir .

# Hailo SDK 가 출력하는 HEF 이름 가변 (모델 내부 name 기준)
for guess in psp_mb.hef psp_mb_3d.hef psp_mb_2d.hef psp_mb4.hef; do
    if [ -f "$guess" ] && [ "$guess" != "${OUT}.hef" ]; then
        mv "$guess" "${OUT}.hef"
        break
    fi
done

if [ -f ${OUT}.hef ]; then
    echo ""
    echo "[OK] Compiled: ${OUT}.hef  ($(ls -lh ${OUT}.hef | awk '{print $5}'))"
else
    echo "[FAIL] Compilation failed"
    exit 1
fi
