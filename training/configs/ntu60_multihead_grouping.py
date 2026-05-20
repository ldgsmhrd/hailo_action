"""NTU 60 action → 5-head 자동 그룹화 매핑 (보조 실험용).

본 매핑은 의미 기반 자동 분류:
  - 정적/동적 라벨 충돌 해결 규칙을 NTU 표준 라벨에 적용
  - 60 액션 → (상체/하체/자세/손/발 + interaction) 동시 출력

목적: 우리 multi-head 구조가 표준 데이터셋에도 적용 가능함을 입증.
"""

# 5-head 자동 분해 매핑 — 의미 기반 (정적/동적 충돌 해결 규칙 적용)
# 각 NTU 액션을 5개 카테고리 라벨에 동시 할당

# 형식: action_idx (0~59): { 'upper': N, 'lower': N, 'pose': N, 'hand': N, 'foot': N }
# N = 해당 헤드 라벨 (없으면 0 = "none")

# 우선 모든 60 클래스를 5-head 라벨로 매핑한 결과
# (자동 분해의 일례 — 실제 paper 에서는 사람 검증 후 사용)

NTU60_TO_5HEAD = {
    # 일상 동작 - 상체 중심
    0: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},   # drink_water → upper=raise, pose=standing
    1: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},   # eat_meal
    2: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},   # brushing_teeth
    3: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},   # brushing_hair
    4: {'upper': 0, 'lower': 0, 'pose': 5, 'hand': 0, 'foot': 0},   # drop → pose=standing-bending
    5: {'upper': 0, 'lower': 0, 'pose': 5, 'hand': 0, 'foot': 0},   # pickup
    6: {'upper': 2, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},   # throw → upper=wave (motion)

    # 자세 전환
    7: {'upper': 0, 'lower': 0, 'pose': 0, 'hand': 0, 'foot': 0},   # sitting_down → pose=sit (transition)
    8: {'upper': 0, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},   # standing_up
    9: {'upper': 3, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},   # clapping → upper=clap

    # 책상 / 손가락 작업
    10: {'upper': 0, 'lower': 0, 'pose': 0, 'hand': 0, 'foot': 0},  # reading → pose=sit
    11: {'upper': 0, 'lower': 0, 'pose': 0, 'hand': 0, 'foot': 0},  # writing
    12: {'upper': 0, 'lower': 0, 'pose': 0, 'hand': 0, 'foot': 0},  # tear_up_paper

    # 의복
    13: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # wear_jacket
    14: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # take_off_jacket
    15: {'upper': 0, 'lower': 0, 'pose': 5, 'hand': 0, 'foot': 0},  # wear_a_shoe → bending
    16: {'upper': 0, 'lower': 0, 'pose': 5, 'hand': 0, 'foot': 0},  # take_off_a_shoe
    17: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # wear_on_glasses
    18: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # take_off_glasses
    19: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # put_on_hat
    20: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # take_off_hat

    # 응원 / 손짓
    21: {'upper': 2, 'lower': 0, 'pose': 4, 'hand': 2, 'foot': 0},  # cheer_up → hand=raise-both
    22: {'upper': 2, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # hand_waving
    23: {'upper': 0, 'lower': 6, 'pose': 4, 'hand': 0, 'foot': 0},  # kicking_something → lower=kick
    24: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # reach_into_pocket

    # 점프
    25: {'upper': 0, 'lower': 8, 'pose': 4, 'hand': 0, 'foot': 0},  # hopping_one_foot → lower=jump-1leg
    26: {'upper': 0, 'lower': 7, 'pose': 4, 'hand': 0, 'foot': 0},  # jump_up → lower=jump-2feet

    # 전화 / 작업
    27: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # make_phone_call
    28: {'upper': 0, 'lower': 0, 'pose': 0, 'hand': 0, 'foot': 0},  # playing_with_phone (sit assumed)
    29: {'upper': 0, 'lower': 0, 'pose': 0, 'hand': 0, 'foot': 0},  # typing
    30: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # pointing → upper=raise
    31: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # taking_selfie
    32: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # check_watch
    33: {'upper': 3, 'lower': 0, 'pose': 4, 'hand': 2, 'foot': 0},  # rub_two_hands → clap + raise-both
    34: {'upper': 0, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # nod_head_bow
    35: {'upper': 0, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # shake_head
    36: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # wipe_face
    37: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # salute
    38: {'upper': 3, 'lower': 0, 'pose': 4, 'hand': 2, 'foot': 0},  # put_palms_together
    39: {'upper': 0, 'lower': 0, 'pose': 4, 'hand': 1, 'foot': 0},  # cross_hands → hand=cross-arms

    # 의료 / 신체 접촉
    40: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # sneeze_cough
    41: {'upper': 0, 'lower': 1, 'pose': 4, 'hand': 0, 'foot': 0},  # staggering → lower=pacing
    42: {'upper': 0, 'lower': 5, 'pose': 6, 'hand': 0, 'foot': 0},  # falling_down → lower=fall, pose=lying
    43: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # touch_head
    44: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # touch_chest
    45: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # touch_back
    46: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # touch_neck
    47: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # nausea
    48: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # fan_self

    # 상호작용 (2명) — interaction 헤드가 별도 필요할 수도 있음
    49: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # punch_other → upper=punch (single)
    50: {'upper': 0, 'lower': 6, 'pose': 4, 'hand': 0, 'foot': 0},  # kick_other → lower=kick
    51: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # push_other
    52: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # pat_back
    53: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # point_finger
    54: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # hugging
    55: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # giving_object
    56: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # touch_pocket
    57: {'upper': 1, 'lower': 0, 'pose': 4, 'hand': 0, 'foot': 0},  # handshake
    58: {'upper': 0, 'lower': 2, 'pose': 4, 'hand': 0, 'foot': 0},  # walking_towards → lower=walk
    59: {'upper': 0, 'lower': 2, 'pose': 4, 'hand': 0, 'foot': 0},  # walking_apart
}

# 본 매핑은 자동 생성 후 사람 검토 필요.
# 정적/동적 충돌 해결 규칙:
#   - "sitting_down", "standing_up" 같은 transition → pose 헤드만
#   - "punching", "kicking" → upper/lower 동작 헤드
#   - "drink water" 같은 일상 → 상체 동작 + pose=standing
