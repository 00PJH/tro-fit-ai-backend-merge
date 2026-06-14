"""
landmark_extractor.py — MediaPipe Tasks API 랜드마크 추출 레이어
─────────────────────────────────────────────────────────────────────────────

책임:
  - PoseLandmarker 초기화 (모델 로딩, 설정)
  - VIDEO 모드 프레임별 랜드마크 추출 (detect_for_video)
  - 원본 이미지/프레임 위에 랜드마크 시각화 (draw_landmarks_on_frame)
  - 랜드마크 결과 → Dict 스키마 변환 (extract_frame_record)
    - normalized landmarks (x, y, z, visibility, pixel_x, pixel_y)
    - world_landmarks (x, y, z, visibility) — 미터 단위 3D 좌표 포함 ★
    - timestamp_ms, frame_index — 시간축 추적 가능 ★
    - presence 필드 추가 ★

설계 포인트:
  - BlazePoseLandmark SSOT 사용 → LANDMARK_NAMES 하드코딩 배열 사용하지 않음
  - Dict 구조 유지 → angle_engine.get_lm()과 완전 호환
  - context manager (with 구문) → 모델 자원 안전 해제 보장
  - VIDEO 모드 + 타임스탬프 단조 증가 → 프레임 간 추적(Tracking) 최적화
  - 좌/우 색상 분리 + visibility 필터 → 임상적 가독성

의존성:
  from core.landmarks import BlazePoseLandmark, POSE_CONNECTIONS, LEFT_LANDMARKS, RIGHT_LANDMARKS
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from core.landmarks import (
    BlazePoseLandmark,
    LEFT_LANDMARKS,
    POSE_CONNECTIONS,
    RIGHT_LANDMARKS,
)

# ──────────────────────────────────────────────────────────────────────────────
# 상수: 시각화 색상 (BGR)
# ──────────────────────────────────────────────────────────────────────────────
_COLOR_LEFT_PT  = (255, 217,   0)   # 하늘색 — 좌측 관절
_COLOR_RIGHT_PT = (  0, 138, 255)   # 주황색 — 우측 관절
_COLOR_OTHER_PT = (255, 255, 255)   # 흰색   — 코 등 중립 관절
_COLOR_LINE     = (255, 255, 255)   # 흰색 연결선

_VISIBILITY_DRAW_THRESHOLD = 0.2    # 시각화 전용 임계값 (각도 계산 임계값과 별도)


# ──────────────────────────────────────────────────────────────────────────────
# PoseLandmarker 생성
# ──────────────────────────────────────────────────────────────────────────────
def create_landmarker(
    model_path: str | Path,
    num_poses:  int   = 1,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence:  float = 0.5,
    min_tracking_confidence:       float = 0.5,
) -> vision.PoseLandmarker:
    """
    VIDEO 모드 PoseLandmarker를 생성하고 반환합니다.

    VIDEO 모드는 프레임 간 추적(Tracking)을 활성화하여
    IMAGE 모드 대비 실시간 처리 성능이 우수합니다 (10.48 ms / 95.42 FPS 실측).

    반드시 with 구문과 함께 사용하여 자원을 안전하게 해제하세요:
        with create_landmarker(model_path) as landmarker:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

    Args:
        model_path: pose_landmarker_full.task 파일 경로
        num_poses:  동시에 감지할 최대 포즈 수 (기본값: 1)
        min_pose_detection_confidence: 포즈 감지 최소 신뢰도
        min_pose_presence_confidence:  포즈 존재 최소 신뢰도
        min_tracking_confidence:       추적 최소 신뢰도

    Returns:
        vision.PoseLandmarker (context manager 지원)

    Raises:
        FileNotFoundError: 모델 파일이 없을 때
        RuntimeError:      MediaPipe 초기화 실패 시
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"[ERROR] 모델 파일을 찾을 수 없습니다: {model_path}\n"
            "  → 프로젝트 루트의 models/ 폴더에 pose_landmarker_full.task 를 배치하세요.\n"
            "  다운로드 명령 (PowerShell):\n"
            "  Invoke-WebRequest -Uri \"https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task\" "
            "-OutFile \"models/pose_landmarker_full.task\""
        )

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def create_image_landmarker(
    model_path: str | Path,
    num_poses:  int   = 1,
    min_pose_detection_confidence: float = 0.3,
    min_pose_presence_confidence:  float = 0.3,
) -> vision.PoseLandmarker:
    """
    IMAGE 모드 PoseLandmarker를 생성하고 반환합니다.

    IMAGE 모드는 각 프레임을 독립적으로 분석합니다 (프레임 간 추적 없음).
    스냅샷 ROM 측정처럼 특정 타임스탬프의 정지 이미지를 분석할 때 사용하세요.

    VIDEO 모드와의 차이:
      - IMAGE: 독립 프레임 분석, detect() 사용, 타임스탬프 불필요
      - VIDEO: 순차 프레임 추적, detect_for_video() + 단조증가 타임스탬프 필수

    반드시 with 구문으로 사용하세요:
        with create_image_landmarker(model_path) as lm:
            record, raw = analyze_single_frame(lm, frame_bgr)

    Args:
        model_path: pose_landmarker_full.task 파일 경로
        num_poses:  동시에 감지할 최대 포즈 수 (기본값: 1)
        min_pose_detection_confidence: 포즈 감지 최소 신뢰도 (기본값: 0.3, 스냅샷용 완화)
        min_pose_presence_confidence:  포즈 존재 최소 신뢰도 (기본값: 0.3)

    Returns:
        vision.PoseLandmarker (context manager 지원)

    Raises:
        FileNotFoundError: 모델 파일이 없을 때
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"[ERROR] 모델 파일을 찾을 수 없습니다: {model_path}\n"
            "  → 프로젝트 루트의 models/ 폴더에 pose_landmarker_full.task 를 배치하세요.\n"
            "  다운로드 명령 (PowerShell):\n"
            "  Invoke-WebRequest -Uri \"https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task\" "
            "-OutFile \"models/pose_landmarker_full.task\""
        )

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def analyze_single_frame(
    landmarker:   vision.PoseLandmarker,
    frame_bgr:    np.ndarray,
    frame_index:  int = 0,
    timestamp_ms: int = 0,
) -> tuple[dict, Any]:
    """
    IMAGE 모드 PoseLandmarker로 단일 프레임을 분석합니다.

    create_image_landmarker()로 생성한 랜드마커에서만 사용하세요.
    VIDEO 모드 랜드마커에서 호출하면 RuntimeError 발생합니다.

    Args:
        landmarker:   create_image_landmarker()로 생성한 PoseLandmarker
        frame_bgr:    OpenCV BGR 이미지 (np.ndarray)
        frame_index:  비디오에서의 실제 프레임 번호 (기록용)
        timestamp_ms: 비디오 타임스탬프 ms (기록용, 추론에는 영향 없음)

    Returns:
        tuple[dict, PoseLandmarkerResult]:
          - [0] frame_record: extract_frame_record()와 동일한 Dict 스키마
          - [1] raw_result:   draw_landmarks_on_frame() 등 시각화에 사용 가능한 원본 결과
    """
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    raw_result = landmarker.detect(mp_image)  # IMAGE 모드: detect() (타임스탬프 불필요)
    frame_record = extract_frame_record(raw_result, frame_index, timestamp_ms, w, h)
    return frame_record, raw_result




# ──────────────────────────────────────────────────────────────────────────────
# 랜드마크 추출: MediaPipe 결과 → Dict 스키마
# ──────────────────────────────────────────────────────────────────────────────
def _optional_float(obj: Any, attr: str) -> float | None:
    """객체에 해당 속성이 있으면 float 반환, 없으면 None."""
    val = getattr(obj, attr, None)
    return float(val) if val is not None else None


def extract_frame_record(
    result:       vision.PoseLandmarkerResult,
    frame_index:  int,
    timestamp_ms: int,
    width:        int,
    height:       int,
) -> dict:
    """
    MediaPipe PoseLandmarkerResult → Dict 스키마 변환.

    출력 스키마 (angle_engine과 완전 호환):
    {
        "frame_index":  int,           # 시간축 추적 ★
        "timestamp_ms": int,           # 동작 구간 분석 ★
        "detected":     bool,
        "num_poses":    int,
        "image_width":  int,
        "image_height": int,
        "poses": [
            {
                "pose_index": int,
                "landmarks": {                 # Dict 구조 (이름 Key) — O(1) 접근
                    "left_shoulder": {
                        "x": float,            # 정규화 좌표 (0~1)
                        "y": float,
                        "z": float,
                        "visibility": float,
                        "presence":   float | None,   # presence 추가 ★
                        "pixel_x":    int,     # 사전계산 픽셀 좌표
                        "pixel_y":    int,
                    },
                    ...
                },
                "world_landmarks": {           # 미터 단위 3D 좌표 ★
                    "left_shoulder": {
                        "x": float,            # 미터 단위 (카메라 거리 영향 제거)
                        "y": float,
                        "z": float,
                        "visibility": float,
                        "presence":   float | None,
                    },
                    ...
                }
            }
        ]
    }

    Args:
        result:       PoseLandmarker.detect_for_video() 반환값
        frame_index:  현재 프레임 번호 (0-indexed)
        timestamp_ms: 비디오 타임스탬프 (ms)
        width:        프레임 너비 (pixel)
        height:       프레임 높이 (pixel)

    Returns:
        dict: 위 스키마를 따르는 딕셔너리
    """
    detected = bool(result.pose_landmarks)
    num_poses = len(result.pose_landmarks) if detected else 0

    poses = []
    for pose_idx in range(num_poses):
        pose_landmarks  = result.pose_landmarks[pose_idx]
        world_landmarks = (
            result.pose_world_landmarks[pose_idx]
            if result.pose_world_landmarks and pose_idx < len(result.pose_world_landmarks)
            else []
        )

        # ── 정규화 좌표 (normalized) ─────────────────────────────────
        lm_dict: dict[str, dict] = {}
        for idx, lm in enumerate(pose_landmarks):
            try:
                key = BlazePoseLandmark(idx).json_key()  # SSOT O(1)
            except ValueError:
                key = f"landmark_{idx}"                   # 범위 밖 방어

            lm_dict[key] = {
                "x":          round(float(lm.x), 6),
                "y":          round(float(lm.y), 6),
                "z":          round(float(lm.z), 6),
                "visibility": round(float(lm.visibility), 6),
                "presence":   _optional_float(lm, "presence"),
                "pixel_x":    int(lm.x * width),
                "pixel_y":    int(lm.y * height),
            }

        # ── world 좌표 (미터 단위) ────────────────────────────────────
        world_dict: dict[str, dict] = {}
        for idx, wlm in enumerate(world_landmarks):
            try:
                key = BlazePoseLandmark(idx).json_key()
            except ValueError:
                key = f"landmark_{idx}"

            world_dict[key] = {
                "x":          round(float(wlm.x), 6),
                "y":          round(float(wlm.y), 6),
                "z":          round(float(wlm.z), 6),
                "visibility": round(float(wlm.visibility), 6),
                "presence":   _optional_float(wlm, "presence"),
            }

        poses.append({
            "pose_index":      pose_idx,
            "landmarks":       lm_dict,
            "world_landmarks": world_dict,
        })

    return {
        "frame_index":  frame_index,
        "timestamp_ms": timestamp_ms,
        "detected":     detected,
        "num_poses":    num_poses,
        "image_width":  width,
        "image_height": height,
        "poses":        poses,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 시각화: 원본 프레임 위에 랜드마크 오버레이
# ──────────────────────────────────────────────────────────────────────────────
def draw_landmarks_on_frame(
    frame:              np.ndarray,
    detection_result:   vision.PoseLandmarkerResult,
    visibility_threshold: float = _VISIBILITY_DRAW_THRESHOLD,
) -> np.ndarray:
    """
    OpenCV BGR 프레임 위에 포즈 랜드마크와 연결선을 고품질로 렌더링합니다.

    특징:
      - LINE_AA 안티앨리어싱 → 매끄러운 선
      - 좌/우 색상 분리 → 임상 환경에서 좌/우 식별 용이
      - 흰색 테두리 원 → 어두운/밝은 배경 모두 가시성 확보
      - visibility 필터 → 신뢰도 낮은 관절은 표시하지 않음
      - 원본 프레임 보존 (np.copy)

    Args:
        frame:                원본 BGR 이미지
        detection_result:     PoseLandmarker 추론 결과
        visibility_threshold: 이 미만 visibility 관절은 그리지 않음

    Returns:
        np.ndarray: 랜드마크가 오버레이된 BGR 이미지
    """
    annotated = np.copy(frame)
    h, w = annotated.shape[:2]

    if not detection_result.pose_landmarks:
        return annotated

    for pose_landmarks in detection_result.pose_landmarks:
        coords: dict[int, tuple[int, int]] = {}

        # 1. 픽셀 좌표 구성 (visibility 필터)
        for idx, lm in enumerate(pose_landmarks):
            if lm.visibility >= visibility_threshold:
                coords[idx] = (int(lm.x * w), int(lm.y * h))

        # 2. POSE_CONNECTIONS 기반 뼈대 연결선 (흰색, LINE_AA)
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx in coords and end_idx in coords:
                cv2.line(
                    annotated,
                    coords[start_idx],
                    coords[end_idx],
                    _COLOR_LINE,
                    2,
                    lineType=cv2.LINE_AA,
                )

        # 3. 관절 포인트 (흰색 외곽 + 좌/우 색상 내부원)
        for idx, pt in coords.items():
            cv2.circle(annotated, pt, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

            bpl = BlazePoseLandmark(idx) if idx <= 32 else None
            if bpl in LEFT_LANDMARKS:
                inner_color = _COLOR_LEFT_PT
            elif bpl in RIGHT_LANDMARKS:
                inner_color = _COLOR_RIGHT_PT
            else:
                inner_color = _COLOR_OTHER_PT

            cv2.circle(annotated, pt, 2, inner_color, -1, lineType=cv2.LINE_AA)

    return annotated


# ──────────────────────────────────────────────────────────────────────────────
# 실시간 HUD: FPS / Latency 오버레이
# ──────────────────────────────────────────────────────────────────────────────
def draw_performance_hud(
    frame:       np.ndarray,
    avg_fps:     float,
    latency_ms:  float,
    avg_latency: float,
    title:       str = "Tro-Fit | ROM Pipeline",
) -> None:
    """
    화면 좌상단에 실시간 성능 지표(FPS, Latency, 부하 상태)를 HUD 스타일로 오버레이합니다.

    색상 분기:
      - FPS    : ≥25 초록 / ≥15 노랑 / 미만 빨강
      - Latency: <35ms 초록 / <70ms 노랑 / 이상 빨강
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (460, 135), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, title, (18, 32), font, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    fps_color = (
        (50, 230, 50)  if avg_fps >= 25 else
        (50, 230, 230) if avg_fps >= 15 else
        (50,  50, 230)
    )
    cv2.putText(
        frame, f"Avg FPS    : {avg_fps:6.1f} FPS",
        (18, 60), font, 0.58, fps_color, 2, cv2.LINE_AA,
    )

    lat_color = (
        (50, 230, 50)  if avg_latency < 35 else
        (50, 230, 230) if avg_latency < 70 else
        (50,  50, 230)
    )
    cv2.putText(
        frame, f"Latency    : {latency_ms:6.1f} ms  (Avg: {avg_latency:.1f} ms)",
        (18, 88), font, 0.52, lat_color, 1, cv2.LINE_AA,
    )

    load_str = (
        "LIGHT (Optimal)" if avg_latency < 35 else
        "MEDIUM (Normal)" if avg_latency < 70 else
        "HEAVY  (Bottleneck)"
    )
    load_color = (
        (50, 230, 50)  if avg_latency < 35 else
        (50, 200, 200) if avg_latency < 70 else
        (50,  80, 230)
    )
    cv2.putText(
        frame, f"Est. Load  : {load_str}",
        (18, 115), font, 0.52, load_color, 1, cv2.LINE_AA,
    )
