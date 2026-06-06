"""
Tro-Fit Vision AI — 부위별 ROM(관절가동범위) 계산 모듈
=======================================================
MediaPipe Tasks API (PoseLandmarker) 기반 랜드마크 포맷에 맞게 작성되었습니다.

[중요] 랜드마크 포맷 변경 사항 (Tasks API):
    - 구버전 (solutions): results.pose_landmarks.landmark[idx]
    - 신버전 (Tasks API): result.pose_landmarks[0]  → Python list[NormalizedLandmark]
                          즉, frame_landmarks[idx].x / .y / .z 로 직접 접근

각 ROM 함수는 아래 형태의 리스트를 입력으로 받습니다:
    pose_landmarks_list: List[ List[NormalizedLandmark] | None ]
        - 각 원소: 하나의 프레임에서 검출된 33개 랜드마크 리스트
        - 검출 실패 프레임: None

MediaPipe Pose Landmark 인덱스 참조:
    https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
"""
import os
import sys

# ── 절대 경로 기반 임포트 (어느 디렉토리에서 실행해도 동작) ──────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.vision_ai.rom_prototype.angle_calculator import calculate_angle_3d

# ── MediaPipe Pose Landmark 인덱스 상수 ────────────────────────────
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_KNEE      = 25
RIGHT_KNEE     = 26
LEFT_ANKLE     = 27
RIGHT_ANKLE    = 28


def extract_landmark_coords(landmark) -> list[float]:
    """
    NormalizedLandmark 객체(또는 x, y, z 속성을 가진 객체)로부터
    [x, y, z] 리스트를 반환합니다.

    Tasks API 랜드마크: result.pose_landmarks[0][idx] — 직접 .x/.y/.z 접근
    """
    return [float(landmark.x), float(landmark.y), float(landmark.z)]


def calculate_knee_rom(pose_landmarks_list: list, side: str = "left") -> dict:
    """
    프레임별 랜드마크 시퀀스를 분석하여 무릎 관절가동범위(ROM)를 계산합니다.
    무릎 각도 정의: 엉덩이(Hip) — 무릎(Knee) — 발목(Ankle)

    Args:
        pose_landmarks_list: List[ List[NormalizedLandmark] | None ]
            (각 원소: Tasks API의 result.pose_landmarks[0], 또는 None)
        side: 'left' 또는 'right'

    Returns:
        dict: { "min", "max", "rom", "all_angles" }
    """
    hip_idx   = LEFT_HIP   if side == "left" else RIGHT_HIP
    knee_idx  = LEFT_KNEE  if side == "left" else RIGHT_KNEE
    ankle_idx = LEFT_ANKLE if side == "left" else RIGHT_ANKLE

    angles = []
    for frame_landmarks in pose_landmarks_list:
        if frame_landmarks is None:
            continue
        try:
            hip   = extract_landmark_coords(frame_landmarks[hip_idx])
            knee  = extract_landmark_coords(frame_landmarks[knee_idx])
            ankle = extract_landmark_coords(frame_landmarks[ankle_idx])
            angle = calculate_angle_3d(hip, knee, ankle)
            angles.append(angle)
        except (IndexError, AttributeError, TypeError):
            continue

    if not angles:
        return {"min": 0.0, "max": 0.0, "rom": 0.0, "all_angles": []}

    min_angle = float(min(angles))
    max_angle = float(max(angles))
    # 가동 범위 = 완전히 폈을 때(최대 각도) - 가장 굽혔을 때(최소 각도)
    return {
        "min": min_angle,
        "max": max_angle,
        "rom": max_angle - min_angle,
        "all_angles": angles
    }


def calculate_shoulder_rom(pose_landmarks_list: list, side: str = "left") -> dict:
    """
    프레임별 랜드마크 시퀀스를 분석하여 어깨 관절가동범위(ROM)를 계산합니다.
    어깨 각도 정의: 팔꿈치(Elbow) — 어깨(Shoulder) — 엉덩이(Hip)

    Args:
        pose_landmarks_list: List[ List[NormalizedLandmark] | None ]
        side: 'left' 또는 'right'

    Returns:
        dict: { "min", "max", "rom", "all_angles" }
    """
    elbow_idx    = LEFT_ELBOW    if side == "left" else RIGHT_ELBOW
    shoulder_idx = LEFT_SHOULDER if side == "left" else RIGHT_SHOULDER
    hip_idx      = LEFT_HIP      if side == "left" else RIGHT_HIP

    angles = []
    for frame_landmarks in pose_landmarks_list:
        if frame_landmarks is None:
            continue
        try:
            elbow    = extract_landmark_coords(frame_landmarks[elbow_idx])
            shoulder = extract_landmark_coords(frame_landmarks[shoulder_idx])
            hip      = extract_landmark_coords(frame_landmarks[hip_idx])
            angle    = calculate_angle_3d(elbow, shoulder, hip)
            angles.append(angle)
        except (IndexError, AttributeError, TypeError):
            continue

    if not angles:
        return {"min": 0.0, "max": 0.0, "rom": 0.0, "all_angles": []}

    min_angle = float(min(angles))
    max_angle = float(max(angles))
    return {
        "min": min_angle,
        "max": max_angle,
        "rom": max_angle - min_angle,
        "all_angles": angles
    }


def calculate_spine_flexion_rom(pose_landmarks_list: list) -> dict:
    """
    프레임별 랜드마크 시퀀스를 분석하여 척추 굽힘(Spine Flexion) ROM을 계산합니다.

    간이 척추 각도 정의:
        어깨 중점(11,12 평균) — 골반 중점(23,24 평균) — 무릎 중점(25,26 평균)
        → 일어선 자세: ~180도 (일직선)
        → 허리 굽힘:  각도 감소

    Args:
        pose_landmarks_list: List[ List[NormalizedLandmark] | None ]

    Returns:
        dict: { "min", "max", "rom", "all_angles" }
    """
    angles = []

    for frame_landmarks in pose_landmarks_list:
        if frame_landmarks is None:
            continue
        try:
            # 좌우 어깨 중점
            l_sh = extract_landmark_coords(frame_landmarks[LEFT_SHOULDER])
            r_sh = extract_landmark_coords(frame_landmarks[RIGHT_SHOULDER])
            shoulder_mid = [(l_sh[i] + r_sh[i]) / 2 for i in range(3)]

            # 좌우 골반 중점
            l_hip = extract_landmark_coords(frame_landmarks[LEFT_HIP])
            r_hip = extract_landmark_coords(frame_landmarks[RIGHT_HIP])
            hip_mid = [(l_hip[i] + r_hip[i]) / 2 for i in range(3)]

            # 좌우 무릎 중점
            l_knee = extract_landmark_coords(frame_landmarks[LEFT_KNEE])
            r_knee = extract_landmark_coords(frame_landmarks[RIGHT_KNEE])
            knee_mid = [(l_knee[i] + r_knee[i]) / 2 for i in range(3)]

            # 척추 각도: 어깨중점 - 골반중점(기준) - 무릎중점
            angle = calculate_angle_3d(shoulder_mid, hip_mid, knee_mid)
            angles.append(angle)
        except (IndexError, AttributeError, TypeError):
            continue

    if not angles:
        return {"min": 0.0, "max": 0.0, "rom": 0.0, "all_angles": []}

    min_angle = float(min(angles))
    max_angle = float(max(angles))
    # 굽힌 가동 범위 = 선 자세(최대) - 가장 깊이 굽힌 자세(최소)
    return {
        "min": min_angle,
        "max": max_angle,
        "rom": max_angle - min_angle,
        "all_angles": angles
    }
