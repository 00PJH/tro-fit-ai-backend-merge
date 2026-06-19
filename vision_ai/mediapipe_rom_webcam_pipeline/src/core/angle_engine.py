"""
angle_engine.py — 3D 관절 각도 계산 엔진 (ROM Pipeline 핵심)
─────────────────────────────────────────────────────────────────────────────

책임:
  - 3D 벡터 내적 기반 각도 계산 (수치 안정성 보장)
  - 관절별 각도 계산 함수 (무릎, 팔꿈치, 어깨, 고관절, 발목)
  - visibility 신뢰도 게이트 (_compute) — 미달 시 reliable=False로 명시적 표시
  - 포즈 전체 분석 (analyze_pose)

설계 원칙:
  - 이 모듈은 순수 계산 레이어 → I/O 없음, 외부 의존성 없음 (numpy만 허용)
  - BlazePoseLandmark SSOT를 통해 관절 인덱싱 (매직 넘버 금지)
  - world_landmarks 지원: 정규화 좌표(x,y,z)와 world 좌표(미터 단위) 모두 처리 가능

각도 정의:
  - 무릎 (Knee Flexion)           : Hip → Knee → Ankle
  - 팔꿈치 (Elbow Flexion)        : Shoulder → Elbow → Wrist
  - 어깨 (Shoulder Abduction)     : Elbow → Shoulder → Hip
  - 고관절 (Hip Flexion)          : Shoulder → Hip → Knee
  - 발목 (Ankle Dorsiflexion)     : Knee → Ankle → Foot Index

의존성:
  from core.landmarks import BlazePoseLandmark as BPL
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import NamedTuple

import numpy as np

from core.landmarks import BlazePoseLandmark as BPL


# ──────────────────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────────────────
VISIBILITY_THRESHOLD: float = 0.65  # 이 미만 visibility → 신뢰 불가 (임상 기준)


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────────────────────────────────────
class LandmarkPoint(NamedTuple):
    """단일 랜드마크의 3D 좌표와 신뢰도."""
    x: float
    y: float
    z: float
    visibility: float
    name: str


@dataclass
class JointAngleResult:
    """관절 하나의 각도 계산 결과."""
    joint:        str
    angle_deg:    float | None   # None = visibility 미달
    reliable:     bool
    point_a:      str   = ""
    vertex:       str   = ""
    point_c:      str   = ""
    visibility_a: float = 0.0
    visibility_v: float = 0.0
    visibility_c: float = 0.0

    def display(self) -> str:
        if not self.reliable:
            return (
                f"  [{self.joint:25s}] UNRELIABLE "
                f"(vis: {self.visibility_a:.2f} / {self.visibility_v:.2f} / {self.visibility_c:.2f})"
            )
        return (
            f"  [{self.joint:25s}] {self.angle_deg:7.2f} deg  "
            f"({self.point_a} -- {self.vertex} -- {self.point_c})"
        )


@dataclass
class PoseAngleReport:
    """한 프레임/포즈의 전체 관절 각도 리포트."""
    pose_index:   int
    frame_index:  int  = -1           # 비디오 프레임 인덱스 (-1: 이미지 모드)
    timestamp_ms: int  = -1           # 비디오 타임스탬프 ms (-1: 이미지 모드)
    joints: list[JointAngleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pose_index":   self.pose_index,
            "frame_index":  self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "joints":       [asdict(j) for j in self.joints],
        }

    def reliable_joints(self) -> list[JointAngleResult]:
        """신뢰할 수 있는 관절 각도만 반환합니다."""
        return [j for j in self.joints if j.reliable]


# ──────────────────────────────────────────────────────────────────────────────
# 핵심 수학 엔진
# ──────────────────────────────────────────────────────────────────────────────
def calculate_angle_3d(
    point_a: LandmarkPoint,
    vertex:  LandmarkPoint,
    point_c: LandmarkPoint,
) -> float:
    """
    3D 벡터 내적으로 vertex를 꼭짓점으로 하는 A-Vertex-C 사이각(도)을 반환합니다.

    수치 안정성:
      - 동일 좌표 방어 (norm < 1e-9 → 0.0 반환)
      - arccos 도메인 클리핑 [-1, 1] → NaN/ValueError 원천 차단
      - float64 강제 → float32 누적 오차 방지

    Args:
        point_a: 첫 번째 참조점
        vertex:  꼭짓점 (각도를 측정할 관절)
        point_c: 두 번째 참조점

    Returns:
        float: 각도 (0 ~ 180도), 계산 불가 시 0.0
    """
    a = np.array([point_a.x, point_a.y, point_a.z], dtype=np.float64)
    v = np.array([vertex.x,  vertex.y,  vertex.z],  dtype=np.float64)
    c = np.array([point_c.x, point_c.y, point_c.z], dtype=np.float64)

    va, vc = a - v, c - v
    n_va, n_vc = np.linalg.norm(va), np.linalg.norm(vc)

    if n_va < 1e-9 or n_vc < 1e-9:
        return 0.0

    cos_a = float(np.clip(np.dot(va, vc) / (n_va * n_vc), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


# ──────────────────────────────────────────────────────────────────────────────
# 랜드마크 파싱
# ──────────────────────────────────────────────────────────────────────────────
def _parse(raw: dict, name: str) -> LandmarkPoint:
    """raw dict → LandmarkPoint NamedTuple."""
    return LandmarkPoint(
        x=float(raw["x"]),
        y=float(raw["y"]),
        z=float(raw["z"]),
        visibility=float(raw.get("visibility", 1.0)),
        name=name,
    )


def get_lm(landmarks: dict[str, dict], lm: BPL) -> LandmarkPoint:
    """
    BlazePoseLandmark enum으로 랜드마크를 가져옵니다 (SSOT 연동).

    Args:
        landmarks: extract_landmarks_json()이 생성한 Dict 구조
        lm:        BlazePoseLandmark enum 값

    Returns:
        LandmarkPoint

    Raises:
        KeyError: 해당 관절이 landmarks에 없을 때
    """
    key = lm.json_key()
    return _parse(landmarks[key], key)


# ──────────────────────────────────────────────────────────────────────────────
# 가시성 게이트 (Visibility Gate)
# ──────────────────────────────────────────────────────────────────────────────
def _compute(
    joint_name: str,
    lm_a: LandmarkPoint,
    lm_v: LandmarkPoint,
    lm_c: LandmarkPoint,
    threshold: float = VISIBILITY_THRESHOLD,
) -> JointAngleResult:
    """
    visibility 검증 후 각도 계산. 3점 중 최소값이 threshold 미달 시 reliable=False 반환.

    임상/피트니스 앱에서 신뢰 불가 각도를 UI에 표시하면 안 되므로
    이 게이트는 파이프라인 전체에서 가장 중요한 안전 장치입니다.
    """
    res = JointAngleResult(
        joint=joint_name,
        angle_deg=None,
        reliable=False,
        point_a=lm_a.name,
        vertex=lm_v.name,
        point_c=lm_c.name,
        visibility_a=lm_a.visibility,
        visibility_v=lm_v.visibility,
        visibility_c=lm_c.visibility,
    )
    if min(lm_a.visibility, lm_v.visibility, lm_c.visibility) < threshold:
        return res
    res.reliable  = True
    res.angle_deg = calculate_angle_3d(lm_a, lm_v, lm_c)
    return res


# ──────────────────────────────────────────────────────────────────────────────
# 관절별 각도 계산 함수
# ──────────────────────────────────────────────────────────────────────────────
def compute_knee_angles(
    landmarks: dict,
    threshold: float = VISIBILITY_THRESHOLD,
) -> list[JointAngleResult]:
    """무릎 굴곡: Hip — Knee — Ankle (좌우)"""
    return [
        _compute(
            f"{side}_knee",
            get_lm(landmarks, hip),
            get_lm(landmarks, knee),
            get_lm(landmarks, ankle),
            threshold,
        )
        for side, hip, knee, ankle in [
            ("left",  BPL.LEFT_HIP,  BPL.LEFT_KNEE,  BPL.LEFT_ANKLE),
            ("right", BPL.RIGHT_HIP, BPL.RIGHT_KNEE, BPL.RIGHT_ANKLE),
        ]
    ]


def compute_elbow_angles(
    landmarks: dict,
    threshold: float = VISIBILITY_THRESHOLD,
) -> list[JointAngleResult]:
    """팔꿈치 굴곡: Shoulder — Elbow — Wrist (좌우)"""
    return [
        _compute(
            f"{side}_elbow",
            get_lm(landmarks, shoulder),
            get_lm(landmarks, elbow),
            get_lm(landmarks, wrist),
            threshold,
        )
        for side, shoulder, elbow, wrist in [
            ("left",  BPL.LEFT_SHOULDER,  BPL.LEFT_ELBOW,  BPL.LEFT_WRIST),
            ("right", BPL.RIGHT_SHOULDER, BPL.RIGHT_ELBOW, BPL.RIGHT_WRIST),
        ]
    ]


def compute_shoulder_angles(
    landmarks: dict,
    threshold: float = VISIBILITY_THRESHOLD,
) -> list[JointAngleResult]:
    """어깨 외전/거상: Elbow — Shoulder — Hip (좌우)"""
    return [
        _compute(
            f"{side}_shoulder",
            get_lm(landmarks, elbow),
            get_lm(landmarks, shoulder),
            get_lm(landmarks, hip),
            threshold,
        )
        for side, elbow, shoulder, hip in [
            ("left",  BPL.LEFT_ELBOW,  BPL.LEFT_SHOULDER,  BPL.LEFT_HIP),
            ("right", BPL.RIGHT_ELBOW, BPL.RIGHT_SHOULDER, BPL.RIGHT_HIP),
        ]
    ]


def compute_hip_angles(
    landmarks: dict,
    threshold: float = VISIBILITY_THRESHOLD,
) -> list[JointAngleResult]:
    """고관절 굴곡: Shoulder — Hip — Knee (좌우)"""
    return [
        _compute(
            f"{side}_hip",
            get_lm(landmarks, shoulder),
            get_lm(landmarks, hip),
            get_lm(landmarks, knee),
            threshold,
        )
        for side, shoulder, hip, knee in [
            ("left",  BPL.LEFT_SHOULDER,  BPL.LEFT_HIP,  BPL.LEFT_KNEE),
            ("right", BPL.RIGHT_SHOULDER, BPL.RIGHT_HIP, BPL.RIGHT_KNEE),
        ]
    ]


def compute_ankle_angles(
    landmarks: dict,
    threshold: float = VISIBILITY_THRESHOLD,
) -> list[JointAngleResult]:
    """발목 배측굴곡: Knee — Ankle — Foot Index (좌우)"""
    return [
        _compute(
            f"{side}_ankle",
            get_lm(landmarks, knee),
            get_lm(landmarks, ankle),
            get_lm(landmarks, foot_index),
            threshold,
        )
        for side, knee, ankle, foot_index in [
            ("left",  BPL.LEFT_KNEE,  BPL.LEFT_ANKLE,  BPL.LEFT_FOOT_INDEX),
            ("right", BPL.RIGHT_KNEE, BPL.RIGHT_ANKLE, BPL.RIGHT_FOOT_INDEX),
        ]
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 포즈 전체 분석
# ──────────────────────────────────────────────────────────────────────────────
def analyze_pose(
    pose: dict,
    frame_index:  int = -1,
    timestamp_ms: int = -1,
    threshold:    float = VISIBILITY_THRESHOLD,
    use_world:    bool  = False,
) -> PoseAngleReport:
    """
    단일 포즈 딕셔너리 → PoseAngleReport.

    Args:
        pose:         extract_frame_record()가 생성한 포즈 딕셔너리
        frame_index:  비디오 프레임 인덱스 (이미지 모드: -1)
        timestamp_ms: 비디오 타임스탬프 ms (이미지 모드: -1)
        threshold:    visibility 임계값
        use_world:    True → world_landmarks 기반 각도 계산 (미터 단위, 더 정확)
                      False → normalized landmarks 기반 (기본값)

    Returns:
        PoseAngleReport
    """
    # world_landmarks가 있고 use_world=True면 world 좌표 우선 사용
    if use_world and "world_landmarks" in pose and pose["world_landmarks"]:
        lm = pose["world_landmarks"]
    else:
        lm = pose["landmarks"]

    report = PoseAngleReport(
        pose_index=pose.get("pose_index", 0),
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
    )
    report.joints.extend(compute_knee_angles(lm, threshold))
    report.joints.extend(compute_elbow_angles(lm, threshold))
    report.joints.extend(compute_shoulder_angles(lm, threshold))
    report.joints.extend(compute_hip_angles(lm, threshold))
    report.joints.extend(compute_ankle_angles(lm, threshold))
    return report
