"""
landmarks.py — BlazePose 33 랜드마크 단일 진실 공급원 (Single Source of Truth)
─────────────────────────────────────────────────────────────────────────────

이 파일은 프로젝트 전체에서 사용하는 유일한 랜드마크 정의 소스입니다.
변경이 필요할 경우 이 파일만 수정하면 전체 파이프라인에 즉시 반영됩니다.

IntEnum 사용 이유:
  1. 정방향(idx → name): BlazePoseLandmark(11).name.lower() → "left_shoulder" — O(1)
  2. 역방향(name → idx): BlazePoseLandmark["LEFT_SHOULDER"]  → 11             — O(1)
  3. int 처럼 사용 가능 → 기존 list/tuple 인덱스 코드와 완전 호환
  4. IDE 자동완성 + 오타 시 즉시 AttributeError → 런타임 버그 사전 차단
  5. 이 파일 하나만 수정하면 모든 모듈에 즉시 반영 (DRY 원칙)

사용 예시:
    from core.landmarks import BlazePoseLandmark as BPL, BODY_CONNECTIONS, LEFT_LANDMARKS

    # int 처럼 인덱싱
    idx = BPL.LEFT_SHOULDER          # → 11

    # JSON 키 변환 (소문자 언더스코어)
    key = BPL.LEFT_SHOULDER.name.lower()  # → "left_shoulder"

    # 역방향: index → 이름
    name = BPL(11).name.lower()           # → "left_shoulder"

    # 역방향: 문자열 → enum (대문자로 맞춰야 함)
    lm = BPL["LEFT_SHOULDER"]             # → BlazePoseLandmark.LEFT_SHOULDER
"""

from __future__ import annotations
from enum import IntEnum


# ──────────────────────────────────────────────────────────────────────────────
# BlazePose 33개 랜드마크 정의
# ──────────────────────────────────────────────────────────────────────────────
class BlazePoseLandmark(IntEnum):
    NOSE             = 0
    LEFT_EYE_INNER   = 1
    LEFT_EYE         = 2
    LEFT_EYE_OUTER   = 3
    RIGHT_EYE_INNER  = 4
    RIGHT_EYE        = 5
    RIGHT_EYE_OUTER  = 6
    LEFT_EAR         = 7
    RIGHT_EAR        = 8
    MOUTH_LEFT       = 9
    MOUTH_RIGHT      = 10
    LEFT_SHOULDER    = 11
    RIGHT_SHOULDER   = 12
    LEFT_ELBOW       = 13
    RIGHT_ELBOW      = 14
    LEFT_WRIST       = 15
    RIGHT_WRIST      = 16
    LEFT_PINKY       = 17
    RIGHT_PINKY      = 18
    LEFT_INDEX       = 19
    RIGHT_INDEX      = 20
    LEFT_THUMB       = 21
    RIGHT_THUMB      = 22
    LEFT_HIP         = 23
    RIGHT_HIP        = 24
    LEFT_KNEE        = 25
    RIGHT_KNEE       = 26
    LEFT_ANKLE       = 27
    RIGHT_ANKLE      = 28
    LEFT_HEEL        = 29
    RIGHT_HEEL       = 30
    LEFT_FOOT_INDEX  = 31
    RIGHT_FOOT_INDEX = 32

    def json_key(self) -> str:
        """JSON/dict 키로 사용되는 소문자 언더스코어 형태를 반환합니다."""
        return self.name.lower()


# ──────────────────────────────────────────────────────────────────────────────
# 편의 별칭 (코드 가독성 향상)
# ──────────────────────────────────────────────────────────────────────────────
L = BlazePoseLandmark


# ──────────────────────────────────────────────────────────────────────────────
# 전체 BlazePose 연결 구조 (33개 랜드마크, draw_landmarks_on_image 용)
# ──────────────────────────────────────────────────────────────────────────────
POSE_CONNECTIONS: list[tuple[BlazePoseLandmark, BlazePoseLandmark]] = [
    # 얼굴 윤곽
    (L.NOSE,            L.LEFT_EYE_INNER),
    (L.LEFT_EYE_INNER,  L.LEFT_EYE),
    (L.LEFT_EYE,        L.LEFT_EYE_OUTER),
    (L.LEFT_EYE_OUTER,  L.LEFT_EAR),
    (L.NOSE,            L.RIGHT_EYE_INNER),
    (L.RIGHT_EYE_INNER, L.RIGHT_EYE),
    (L.RIGHT_EYE,       L.RIGHT_EYE_OUTER),
    (L.RIGHT_EYE_OUTER, L.RIGHT_EAR),
    # 입
    (L.MOUTH_LEFT,      L.MOUTH_RIGHT),
    # 어깨
    (L.LEFT_SHOULDER,   L.RIGHT_SHOULDER),
    # 왼팔
    (L.LEFT_SHOULDER,   L.LEFT_ELBOW),
    (L.LEFT_ELBOW,      L.LEFT_WRIST),
    # 오른팔
    (L.RIGHT_SHOULDER,  L.RIGHT_ELBOW),
    (L.RIGHT_ELBOW,     L.RIGHT_WRIST),
    # 왼손
    (L.LEFT_WRIST,      L.LEFT_PINKY),
    (L.LEFT_WRIST,      L.LEFT_INDEX),
    (L.LEFT_WRIST,      L.LEFT_THUMB),
    (L.LEFT_PINKY,      L.LEFT_INDEX),
    # 오른손
    (L.RIGHT_WRIST,     L.RIGHT_PINKY),
    (L.RIGHT_WRIST,     L.RIGHT_INDEX),
    (L.RIGHT_WRIST,     L.RIGHT_THUMB),
    (L.RIGHT_PINKY,     L.RIGHT_INDEX),
    # 몸통
    (L.LEFT_SHOULDER,   L.LEFT_HIP),
    (L.RIGHT_SHOULDER,  L.RIGHT_HIP),
    (L.LEFT_HIP,        L.RIGHT_HIP),
    # 왼다리
    (L.LEFT_HIP,        L.LEFT_KNEE),
    (L.LEFT_KNEE,       L.LEFT_ANKLE),
    (L.LEFT_ANKLE,      L.LEFT_HEEL),
    (L.LEFT_HEEL,       L.LEFT_FOOT_INDEX),
    (L.LEFT_ANKLE,      L.LEFT_FOOT_INDEX),
    # 오른다리
    (L.RIGHT_HIP,       L.RIGHT_KNEE),
    (L.RIGHT_KNEE,      L.RIGHT_ANKLE),
    (L.RIGHT_ANKLE,     L.RIGHT_HEEL),
    (L.RIGHT_HEEL,      L.RIGHT_FOOT_INDEX),
    (L.RIGHT_ANKLE,     L.RIGHT_FOOT_INDEX),
]


# ──────────────────────────────────────────────────────────────────────────────
# 픽토그래픽용 주요 뼈대만 (얼굴 세부·손가락 제외)
# ──────────────────────────────────────────────────────────────────────────────
BODY_CONNECTIONS: list[tuple[BlazePoseLandmark, BlazePoseLandmark]] = [
    (L.LEFT_SHOULDER,  L.RIGHT_SHOULDER),
    (L.LEFT_SHOULDER,  L.LEFT_ELBOW),
    (L.LEFT_ELBOW,     L.LEFT_WRIST),
    (L.RIGHT_SHOULDER, L.RIGHT_ELBOW),
    (L.RIGHT_ELBOW,    L.RIGHT_WRIST),
    (L.LEFT_SHOULDER,  L.LEFT_HIP),
    (L.RIGHT_SHOULDER, L.RIGHT_HIP),
    (L.LEFT_HIP,       L.RIGHT_HIP),
    (L.LEFT_HIP,       L.LEFT_KNEE),
    (L.LEFT_KNEE,      L.LEFT_ANKLE),
    (L.RIGHT_HIP,      L.RIGHT_KNEE),
    (L.RIGHT_KNEE,     L.RIGHT_ANKLE),
    (L.LEFT_ANKLE,     L.LEFT_FOOT_INDEX),
    (L.RIGHT_ANKLE,    L.RIGHT_FOOT_INDEX),
]


# ──────────────────────────────────────────────────────────────────────────────
# 좌/우 랜드마크 분류 (시각화 색상 지정용)
# frozenset: 불변(immutable) → 실수로 수정 불가 + 해시 최적화
# ──────────────────────────────────────────────────────────────────────────────
LEFT_LANDMARKS: frozenset[BlazePoseLandmark] = frozenset({
    L.LEFT_EYE_INNER, L.LEFT_EYE, L.LEFT_EYE_OUTER,
    L.LEFT_EAR, L.MOUTH_LEFT,
    L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST,
    L.LEFT_PINKY, L.LEFT_INDEX, L.LEFT_THUMB,
    L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE,
    L.LEFT_HEEL, L.LEFT_FOOT_INDEX,
})

RIGHT_LANDMARKS: frozenset[BlazePoseLandmark] = frozenset({
    L.RIGHT_EYE_INNER, L.RIGHT_EYE, L.RIGHT_EYE_OUTER,
    L.RIGHT_EAR, L.MOUTH_RIGHT,
    L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST,
    L.RIGHT_PINKY, L.RIGHT_INDEX, L.RIGHT_THUMB,
    L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE,
    L.RIGHT_HEEL, L.RIGHT_FOOT_INDEX,
})
