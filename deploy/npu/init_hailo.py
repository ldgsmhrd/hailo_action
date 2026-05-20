"""
NPU 로드 — 멀티 NPU (etri-board) 와 단일 NPU (Pi5 hailo-8l) 양쪽 지원.

환경변수:
  HAILO_SHARED_NPU=1  → 같은 device 를 여러 프로세스가 공유 (Pi 단일 NPU 시)
                       VDevice scheduler + group_id 사용
"""
import os
from hailo_platform import (
    VDevice, HEF, ConfigureParams,
    HailoStreamInterface, InputVStreamParams, OutputVStreamParams,
    FormatType,
)

try:
    from hailo_platform import HailoSchedulingAlgorithm
    _HAS_SCHEDULER = True
except ImportError:
    _HAS_SCHEDULER = False


def _make_vdevice(device_id):
    """단일 NPU 를 두 프로세스가 공유하려면 scheduler + group_id 필요."""
    shared = os.environ.get('HAILO_SHARED_NPU', '0') == '1'
    if shared and _HAS_SCHEDULER:
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        # group_id 가 같으면 동일 device 공유, multi_process_service 가 중재
        params.group_id = "snvr_shared"
        params.multi_process_service = True
        return VDevice(device_ids=[device_id], params=params)
    return VDevice(device_ids=[device_id])


def _init_single(device_id, hef_path, label):
    vdevice = _make_vdevice(device_id)
    hef = HEF(hef_path)
    cfg_params = ConfigureParams.create_from_hef(
        hef=hef, interface=HailoStreamInterface.PCIe,
    )
    groups = vdevice.configure(hef, cfg_params)
    ng = groups[0]
    ng_params = ng.create_params()

    in_infos = list(hef.get_input_vstream_infos())
    out_infos = list(hef.get_output_vstream_infos())
    in_info = in_infos[0]
    out_info = out_infos[0]                          # 호환성 (action HEF 처럼 단일 출력 모델용)

    in_params = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    print(f"[{label}] device={device_id}  "
          f"in={in_info.name}{tuple(in_info.shape)}  "
          f"outs={len(out_infos)}")
    for vs in out_infos:
        print(f"    · {vs.name}  shape={tuple(vs.shape)}")
    return {
        'vdevice': vdevice,
        'hef': hef,
        'network_group': ng,
        'network_group_params': ng_params,
        'input_info': in_info,
        'output_info': out_info,        # 단일 (action 용)
        'output_infos': out_infos,      # 다중 (pose 9-output 용)
        'input_params': in_params,
        'output_params': out_params,
    }


def init_pose_npu(device_id, npu_index, pose_hef_path):
    """NPU 에 YOLOv8-pose HEF 로드."""
    h = _init_single(device_id, pose_hef_path, f"NPU{npu_index} Pose")
    h['npu_index'] = npu_index
    h['role'] = 'pose'
    return h


def init_action_npu(device_id, npu_index, action_hef_path):
    """NPU 에 ActionResNet18 HEF 로드."""
    h = _init_single(device_id, action_hef_path, f"NPU{npu_index} Action")
    h['npu_index'] = npu_index
    h['role'] = 'action'
    return h
