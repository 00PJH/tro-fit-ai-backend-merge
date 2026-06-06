"""
Tro-Fit Vision AI — ROM 분석 통합 파이프라인 (main_prototype.py)
=================================================================
MediaPipe Tasks API (PoseLandmarker) 기반으로 작성되었습니다.

실행 방법:
    # 프로젝트 루트(c:\\workspace\\trofit)에서 실행
    .\\venv\\Scripts\\python.exe src\\vision_ai\\rom_prototype\\main_prototype.py

    # 비디오 파일 지정 시
    .\\venv\\Scripts\\python.exe src\\vision_ai\\rom_prototype\\main_prototype.py --video path/to/video.mp4

    # 결과 JSON 저장 경로 지정
    .\\venv\\Scripts\\python.exe src\\vision_ai\\rom_prototype\\main_prototype.py --output my_result.json

필수 모델 파일: <프로젝트_루트>/models/pose_landmarker_full.task
모델 다운로드:
    New-Item -ItemType Directory -Force -Path models
    Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task -OutFile models/pose_landmarker_full.task
"""
import os
import sys
import json
import argparse

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── 프로젝트 루트를 sys.path에 추가 (어느 위치에서 실행해도 동작) ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vision_ai.rom_prototype.rom_calculator import (
    calculate_knee_rom,
    calculate_shoulder_rom,
    calculate_spine_flexion_rom,
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pose_landmarker_full.task")


# ──────────────────────────────────────────────────────────────────
# 가상(Mock) 포즈 데이터 생성
# ──────────────────────────────────────────────────────────────────

class _MockLandmark:
    """NormalizedLandmark 구조를 모방하는 간단한 클래스."""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


def create_mock_pose_data() -> list[list[_MockLandmark]]:
    """
    카메라 입력이 없는 개발·CI 환경을 위해,
    무릎을 서서히 굽혔다 펴는 동작과 팔을 올리는 동작을 동시에 시뮬레이션하는
    가상 랜드마크 데이터(100프레임)를 생성합니다.

    반환 형식: List[ List[_MockLandmark] ]
        → Tasks API 의 result.pose_landmarks[0] 와 동일한 구조
    """
    print("[INFO] 테스트용 가상 3D 포즈 랜드마크 데이터를 생성합니다 (100프레임).")
    mock_sequence = []

    for frame_idx in range(100):
        # 사인 파형으로 0.0 ~ 1.0 범위의 t 생성
        t = np.sin(2 * np.pi * frame_idx / 100) * 0.5 + 0.5

        # 33개 None으로 초기화
        lm = [None] * 33

        # ── 엉덩이 (고정) ────────────────────────────────────────
        lm[23] = _MockLandmark(0.48, 0.60, 0.0)   # Left Hip
        lm[24] = _MockLandmark(0.52, 0.60, 0.0)   # Right Hip

        # ── 무릎 (스쿼트 동작 시뮬레이션) ───────────────────────
        knee_y = 0.75 + (t * 0.08)                 # 스쿼트 시 아래로
        lm[25] = _MockLandmark(0.45 - t * 0.03, knee_y, 0.0)  # Left Knee
        lm[26] = _MockLandmark(0.55 + t * 0.03, knee_y, 0.0)  # Right Knee

        # ── 발목 (고정) ──────────────────────────────────────────
        lm[27] = _MockLandmark(0.47, 0.90, 0.0)   # Left Ankle
        lm[28] = _MockLandmark(0.53, 0.90, 0.0)   # Right Ankle

        # ── 어깨 (고정) ──────────────────────────────────────────
        lm[11] = _MockLandmark(0.45, 0.30, 0.0)   # Left Shoulder
        lm[12] = _MockLandmark(0.55, 0.30, 0.0)   # Right Shoulder

        # ── 팔꿈치 (팔 올리기 시뮬레이션) ───────────────────────
        elbow_y = 0.45 - (t * 0.25)               # 팔 올릴수록 Y 감소
        lm[13] = _MockLandmark(0.40 - t * 0.05, elbow_y, 0.0)  # Left Elbow
        lm[14] = _MockLandmark(0.60 + t * 0.05, elbow_y, 0.0)  # Right Elbow

        mock_sequence.append(lm)

    return mock_sequence


# ──────────────────────────────────────────────────────────────────
# 비디오 → 랜드마크 시퀀스 추출 (Tasks API VIDEO 모드)
# ──────────────────────────────────────────────────────────────────

def extract_landmarks_from_video(video_path: str) -> list:
    """
    비디오 파일을 프레임별로 분석하여 랜드마크 시퀀스를 추출합니다.
    Tasks API VIDEO 모드를 사용하여 프레임 간 추적(Tracking)을 활성화합니다.

    Returns:
        List[ List[NormalizedLandmark] | None ]
    """
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 모델 파일 없음: {MODEL_PATH}")
        print("[HINT] 프로젝트 루트에서 아래 명령 실행:")
        print("  New-Item -ItemType Directory -Force -Path models")
        print("  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task -OutFile models/pose_landmarker_full.task")
        sys.exit(1)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    landmarks_sequence = []
    frame_count = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            # VIDEO 모드 타임스탬프: 프레임 번호 기반
            timestamp_ms = int(frame_count * (1000.0 / fps))

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks_sequence.append(result.pose_landmarks[0])  # List[NormalizedLandmark]
            else:
                landmarks_sequence.append(None)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  - {frame_count} 프레임 처리 완료...")

    cap.release()
    detected = sum(1 for x in landmarks_sequence if x is not None)
    rate = (detected / frame_count * 100) if frame_count > 0 else 0
    print(f"[SUCCESS] 총 {frame_count} 프레임 분석 완료 (랜드마크 검출률: {rate:.1f}%)")

    return landmarks_sequence


# ──────────────────────────────────────────────────────────────────
# ROM 파이프라인 통합 실행 함수
# ──────────────────────────────────────────────────────────────────

def run_rom_pipeline(video_path: str = None, output_json_path: str = "rom_analysis_result.json") -> None:
    """
    ROM 분석 핵심 파이프라인 통합 함수.

    1. 비디오 파일 제공 시: Tasks API로 프레임별 랜드마크 추출
    2. 비디오 없을 시: 가상(Mock) 랜드마크 데이터로 알고리즘 검증
    3. ROM 연산 → FALLS Score 산출 → JSON 저장
    """
    # ── 랜드마크 시퀀스 준비 ─────────────────────────────────────
    if video_path and os.path.exists(video_path):
        print(f"[INFO] 비디오 파일 '{video_path}' 분석을 시작합니다...")
        landmarks_sequence = extract_landmarks_from_video(video_path)
    else:
        landmarks_sequence = create_mock_pose_data()

    # ── ROM 연산 ─────────────────────────────────────────────────
    print("[INFO] 관절별 가동범위(ROM) 연산을 시작합니다...")
    knee_left      = calculate_knee_rom(landmarks_sequence, side="left")
    knee_right     = calculate_knee_rom(landmarks_sequence, side="right")
    shoulder_left  = calculate_shoulder_rom(landmarks_sequence, side="left")
    shoulder_right = calculate_shoulder_rom(landmarks_sequence, side="right")
    spine_flexion  = calculate_spine_flexion_rom(landmarks_sequence)

    # ── 좌우 비대칭 판별 (8도 이상 차이 시 비대칭) ──────────────
    knee_asymmetry     = abs(knee_left["rom"] - knee_right["rom"]) > 8.0
    shoulder_asymmetry = abs(shoulder_left["rom"] - shoulder_right["rom"]) > 8.0

    # ── 낙상 위험도 간이 평가 (FALLS Score, 최대 10점) ───────────
    falls_score = 0.0
    if knee_left["rom"] < 75.0 or knee_right["rom"] < 75.0:
        falls_score += 4.5
    if knee_asymmetry:
        falls_score += 3.5
    if spine_flexion["rom"] < 30.0:
        falls_score += 2.0
    falls_score = min(falls_score, 10.0)

    # ── 권장 안무 결정 ────────────────────────────────────────────
    if knee_asymmetry:
        recommendation = (
            "좌우 무릎 각도 편차가 관찰됩니다. "
            "무릎 부담이 적은 쿨다운 스트레칭과 트로트 안무(난이도: 쉬움)를 추천합니다."
        )
    else:
        recommendation = (
            "관절 상태가 양호합니다. "
            "신나는 트로트 안무(난이도: 보통)를 추천합니다."
        )

    # ── 결과 JSON 구성 ────────────────────────────────────────────
    analysis_result = {
        "project_name": "Tro-Fit",
        "version": "1.1",
        "api": "MediaPipe Tasks API — PoseLandmarker (pose_landmarker_full.task)",
        "analysis_type": "FMS_ROM_Evaluation",
        "metrics": {
            "knee_left": {
                "min_flexion_deg":    round(knee_left["min"], 1),
                "max_extension_deg":  round(knee_left["max"], 1),
                "calculated_rom_deg": round(knee_left["rom"], 1),
            },
            "knee_right": {
                "min_flexion_deg":    round(knee_right["min"], 1),
                "max_extension_deg":  round(knee_right["max"], 1),
                "calculated_rom_deg": round(knee_right["rom"], 1),
            },
            "shoulder_left": {
                "min_deg":            round(shoulder_left["min"], 1),
                "max_deg":            round(shoulder_left["max"], 1),
                "calculated_rom_deg": round(shoulder_left["rom"], 1),
            },
            "shoulder_right": {
                "min_deg":            round(shoulder_right["min"], 1),
                "max_deg":            round(shoulder_right["max"], 1),
                "calculated_rom_deg": round(shoulder_right["rom"], 1),
            },
            "spine_flexion": {
                "min_deg":            round(spine_flexion["min"], 1),
                "max_deg":            round(spine_flexion["max"], 1),
                "calculated_rom_deg": round(spine_flexion["rom"], 1),
            },
        },
        "diagnostic_results": {
            "knee_asymmetry_detected":     knee_asymmetry,
            "shoulder_asymmetry_detected": shoulder_asymmetry,
            "falls_risk_score_10":         round(falls_score, 1),
            "recommendation":              recommendation,
        },
    }

    # ── JSON 저장 ─────────────────────────────────────────────────
    out_dir = os.path.dirname(output_json_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=4, ensure_ascii=False)

    # ── 결과 콘솔 출력 ────────────────────────────────────────────
    print("\n" + "=" * 54)
    print("        Tro-Fit ROM 분석 프로토타입 결과")
    print("=" * 54)
    print(f"  Left  Knee ROM     : {knee_left['rom']:.1f} deg")
    print(f"  Right Knee ROM     : {knee_right['rom']:.1f} deg")
    print(f"  Left  Shoulder ROM : {shoulder_left['rom']:.1f} deg")
    print(f"  Right Shoulder ROM : {shoulder_right['rom']:.1f} deg")
    print(f"  Spine Flexion ROM  : {spine_flexion['rom']:.1f} deg")
    print(f"  Knee Asymmetry     : {'DETECTED [!]' if knee_asymmetry else 'Normal [OK]'}")
    print(f"  Shoulder Asymmetry : {'DETECTED [!]' if shoulder_asymmetry else 'Normal [OK]'}")
    print(f"  FALLS Risk Score   : {falls_score:.1f} / 10.0")
    print(f"  Recommendation     : {recommendation}")
    print("=" * 54)
    print(f"[SUCCESS] 최종 분석 결과가 '{output_json_path}'에 저장되었습니다.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tro-Fit ROM Pipeline Prototype")
    parser.add_argument("--video",  type=str, default=None,
                        help="분석할 비디오 파일 경로 (생략 시 모킹 데이터 자동 실행)")
    parser.add_argument("--output", type=str, default="rom_analysis_result.json",
                        help="결과 JSON 저장 경로")
    args = parser.parse_args()

    run_rom_pipeline(args.video, args.output)
