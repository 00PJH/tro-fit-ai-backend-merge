"""
Tro-Fit Vision AI - MediaPipe Tasks API 정적 이미지 벤치마크
============================================================
mediapipe >= 0.10.31 부터 mp.solutions.pose 레거시 API가 제거되었습니다.
이 스크립트는 공식 후속 API인 mediapipe.tasks.python.vision.PoseLandmarker 를 사용합니다.

필수: pose_landmarker_full.task 모델 파일이 아래 경로에 존재해야 합니다.
    <프로젝트_루트>/models/pose_landmarker_full.task

모델 다운로드 명령 (프로젝트 루트에서 실행):
    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -OutFile "models/pose_landmarker_full.task"
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ──────────────────────────────────────────────────────────────────
# 모델 경로 설정 (프로젝트 루트 기준 models/ 폴더)
# ──────────────────────────────────────────────────────────────────
# 이 파일 위치: vision_ai/benchmark/mediapipe_test.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))  # trofit/
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pose_landmarker_full.task")

# ──────────────────────────────────────────────────────────────────
# MediaPipe Pose 33개 관절 연결 정보 (시각화용)
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# ──────────────────────────────────────────────────────────────────
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]


def draw_pose_landmarks(image: np.ndarray, landmarks: list) -> None:
    """
    OpenCV를 사용해 포즈 랜드마크를 이미지 위에 직접 시각화합니다.
    (Tasks API에서는 mp.solutions.drawing_utils 를 사용하지 않습니다.)
    """
    h, w = image.shape[:2]

    # 관절 연결선 (파란색)
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            cv2.line(
                image,
                (int(start_lm.x * w), int(start_lm.y * h)),
                (int(end_lm.x * w), int(end_lm.y * h)),
                (255, 80, 0), 2
            )

    # 관절 포인트 (초록색 원)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 4, (0, 230, 0), -1)


def run_mediapipe_image_benchmark(image_path: str = None, output_dir: str = "results") -> None:
    """
    정적 이미지에 대한 MediaPipe Pose Landmarker (Tasks API) 추론 속도 및 정확도 벤치마크.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 모델 파일 존재 여부 확인 ────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("[HINT] 아래 PowerShell 명령으로 모델을 다운로드하세요 (프로젝트 루트에서 실행):")
        print("  New-Item -ItemType Directory -Force -Path models")
        print("  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task -OutFile models/pose_landmarker_full.task")
        sys.exit(1)

    # ── 1. PoseLandmarker (IMAGE 모드) 초기화 ─────────────────────
    print("[INFO] MediaPipe Tasks API - PoseLandmarker 초기화 중 (IMAGE 모드)...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    print("[INFO] PoseLandmarker 초기화 완료.")

    # ── 2. 이미지 로딩 또는 테스트용 더미 이미지 생성 ──────────────
    if image_path and os.path.exists(image_path):
        print(f"[INFO] '{image_path}' 이미지를 로드합니다.")
        image_bgr = cv2.imread(image_path)
    else:
        print("[INFO] 입력 이미지가 없거나 경로가 유효하지 않아 가상 벤치마크 이미지를 생성합니다.")
        image_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(image_bgr, (320, 150), 40, (255, 255, 255), -1)   # 머리
        cv2.rectangle(image_bgr, (280, 200), (360, 380), (200, 200, 200), -1)  # 몸통
        cv2.line(image_bgr, (280, 220), (220, 300), (180, 180, 180), 15)  # 왼팔
        cv2.line(image_bgr, (360, 220), (420, 300), (180, 180, 180), 15)  # 오른팔
        cv2.line(image_bgr, (300, 380), (280, 470), (180, 180, 180), 20)  # 왼다리
        cv2.line(image_bgr, (340, 380), (360, 470), (180, 180, 180), 20)  # 오른다리

    # Tasks API 는 RGB mp.Image 객체를 요구합니다
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # ── 3. 추론 속도 벤치마크 (웜업 5회, 본 측정 20회) ─────────────
    print("[INFO] 벤치마크 추론을 시작합니다 (웜업 5회, 본 측정 20회)...")

    for _ in range(5):
        landmarker.detect(mp_image)

    latencies = []
    num_runs = 20
    last_result = None

    for i in range(num_runs):
        start = time.perf_counter()
        result = landmarker.detect(mp_image)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        last_result = result
        print(f"  - Run {i+1:02d}: {latency_ms:.2f} ms")

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    print("\n" + "=" * 52)
    print("    MediaPipe Tasks API - Pose Landmarker 벤치마크 결과")
    print("=" * 52)
    print(f"  - 이미지 크기          : {image_bgr.shape[1]}x{image_bgr.shape[0]}")
    print(f"  - 모델 복잡도          : Full (pose_landmarker_full.task)")
    print(f"  - 평균 추론 시간       : {avg_latency:.2f} ms")
    print(f"  - 표준 편차            : {std_latency:.2f} ms")
    print(f"  - 최소 / 최대 시간     : {min_latency:.2f} ms / {max_latency:.2f} ms")
    print(f"  - 초당 처리 프레임(FPS): {1000 / avg_latency:.2f} FPS")
    print("=" * 52 + "\n")

    # ── 4. 결과 시각화 및 저장 ──────────────────────────────────────
    output_image = image_bgr.copy()

    if last_result and last_result.pose_landmarks:
        landmarks = last_result.pose_landmarks[0]  # 첫 번째 검출된 사람의 랜드마크 리스트
        print("[SUCCESS] 신체 랜드마크 검출 성공! 시각화 결과 이미지를 생성합니다.")
        draw_pose_landmarks(output_image, landmarks)

        # 주요 관절 좌표 샘플 출력 (어깨 - 인덱스 11, 12)
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        left_sh = landmarks[LEFT_SHOULDER]
        right_sh = landmarks[RIGHT_SHOULDER]
        print("  - 주요 관절 샘플 (좌우 어깨 좌표):")
        print(f"    * Left Shoulder  (11): x={left_sh.x:.4f}, y={left_sh.y:.4f}, z={left_sh.z:.4f}")
        print(f"    * Right Shoulder (12): x={right_sh.x:.4f}, y={right_sh.y:.4f}, z={right_sh.z:.4f}")
    else:
        print("[WARNING] 신체 랜드마크 검출 실패. (더미 이미지에서는 정상입니다.)")

    output_path = os.path.join(output_dir, "benchmark_result.jpg")
    cv2.imwrite(output_path, output_image)
    print(f"[INFO] 결과 이미지가 '{output_path}'에 저장되었습니다.")

    # ── 5. 리소스 해제 ──────────────────────────────────────────────
    landmarker.close()
    print("[INFO] PoseLandmarker 리소스 해제 완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tro-Fit - MediaPipe Tasks API Pose Landmarker Static Image Benchmark"
    )
    parser.add_argument("--image", type=str, default=None, help="벤치마크할 이미지 경로 (기본값: 더미 이미지 생성)")
    parser.add_argument("--out_dir", type=str, default="results", help="결과물 저장 폴더")
    args = parser.parse_args()

    run_mediapipe_image_benchmark(args.image, args.out_dir)
