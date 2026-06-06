"""
Tro-Fit Vision AI — MediaPipe Tasks API 실시간 웹캠 벤치마크
=============================================================
mediapipe >= 0.10.31 부터 mp.solutions.pose 레거시 API가 제거되었습니다.
이 스크립트는 공식 후속 API인 mediapipe.tasks.python.vision.PoseLandmarker 를 사용합니다.
VIDEO 모드를 사용하여 프레임 간 추적(Tracking)을 활성화하여 실시간 성능을 최적화합니다.

필수: pose_landmarker_full.task 모델 파일이 아래 경로에 존재해야 합니다.
    <프로젝트_루트>/models/pose_landmarker_full.task

모델 다운로드 명령 (프로젝트 루트에서 실행):
    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -OutFile "models/pose_landmarker_full.task"
"""
import os
import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ──────────────────────────────────────────────────────────────────
# 모델 경로 설정 (프로젝트 루트 기준 models/ 폴더)
# ──────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pose_landmarker_full.task")

# ──────────────────────────────────────────────────────────────────
# MediaPipe Pose 33개 관절 연결 정보 (시각화용)
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
    """
    h, w = image.shape[:2]

    # 관절 연결선 (파란색 계열)
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            s = landmarks[start_idx]
            e = landmarks[end_idx]
            cv2.line(
                image,
                (int(s.x * w), int(s.y * h)),
                (int(e.x * w), int(e.y * h)),
                (255, 80, 0), 2
            )

    # 관절 포인트 (초록색 원)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 230, 0), -1)
        cv2.circle(image, (cx, cy), 5, (0, 100, 0), 1)  # 테두리


def draw_metrics_overlay(frame: np.ndarray, avg_fps: float, latency_ms: float, avg_latency: float) -> None:
    """
    화면 좌상단에 실시간 성능 지표를 HUD 스타일로 오버레이합니다.
    """
    # 배경 반투명 박스
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (395, 130), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, "Tro-Fit | MediaPipe Tasks API — PoseLandmarker",
                (18, 32), font, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    # FPS 색상 분기: >=25fps 초록 / >=15fps 노랑 / 미만 빨강
    fps_color = (50, 230, 50) if avg_fps >= 25 else ((50, 230, 230) if avg_fps >= 15 else (50, 50, 230))
    cv2.putText(frame, f"Avg FPS    : {avg_fps:6.1f} FPS",
                (18, 60), font, 0.58, fps_color, 2, cv2.LINE_AA)

    # 레이턴시 색상: <35ms 초록 / <70ms 노랑 / 이상 빨강
    lat_color = (50, 230, 50) if avg_latency < 35 else ((50, 230, 230) if avg_latency < 70 else (50, 50, 230))
    cv2.putText(frame, f"Latency    : {latency_ms:6.1f} ms  (Avg: {avg_latency:.1f} ms)",
                (18, 88), font, 0.52, lat_color, 1, cv2.LINE_AA)

    load_str = "LIGHT (Optimal)" if avg_latency < 35 else ("MEDIUM (Normal)" if avg_latency < 70 else "HEAVY (Bottleneck)")
    load_color = (50, 230, 50) if avg_latency < 35 else ((50, 200, 200) if avg_latency < 70 else (50, 80, 230))
    cv2.putText(frame, f"Est. Load  : {load_str}",
                (18, 115), font, 0.52, load_color, 1, cv2.LINE_AA)


def run_mediapipe_webcam_benchmark() -> None:
    """
    실시간 웹캠 스트림에 대한 MediaPipe PoseLandmarker (Tasks API, VIDEO 모드) 성능 측정.
    VIDEO 모드는 프레임 간 추적(Tracking)을 활용하여 IMAGE 모드보다 실시간 성능이 우수합니다.
    """
    # ── 모델 파일 존재 여부 확인 ────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("[HINT] 아래 PowerShell 명령으로 모델을 다운로드하세요 (프로젝트 루트에서 실행):")
        print("  New-Item -ItemType Directory -Force -Path models")
        print("  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task -OutFile models/pose_landmarker_full.task")
        sys.exit(1)

    # ── 1. PoseLandmarker (VIDEO 모드) 초기화 ─────────────────────
    # VIDEO 모드: 연속 프레임에서 이전 감지 결과를 활용한 추적 최적화
    print("[INFO] MediaPipe Tasks API — PoseLandmarker 초기화 중 (VIDEO 모드)...")
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
    landmarker = vision.PoseLandmarker.create_from_options(options)
    print("[INFO] PoseLandmarker 초기화 완료.\n")

    # ── 2. 웹캠 캡처 인스턴스 생성 ─────────────────────────────────
    print("[INFO] 웹캠을 연결하는 중입니다 (카메라 인덱스: 0)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다. 카메라가 정상적으로 연결되어 있는지 확인하십시오.")
        landmarker.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 카메라 연결 성공: 해상도 {width}x{height}")
    print("[INFO] 'q' 키를 누르면 벤치마크가 안전하게 종료됩니다.\n")

    # ── 3. FPS / Latency 측정 변수 초기화 ─────────────────────────
    prev_time = 0.0
    fps_history: list[float] = []
    latency_history: list[float] = []
    frame_idx = 0  # VIDEO 모드에 전달할 타임스탬프(ms) 계산용

    start_wall = time.perf_counter()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[WARNING] 카메라 프레임을 읽어오는 데 실패했습니다.")
            continue

        # 노인 친화적 UX: 좌우 반전(거울 모드)
        frame = cv2.flip(frame, 1)

        # BGR → RGB 변환 후 MediaPipe Image 객체 생성
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # VIDEO 모드는 타임스탬프(ms)가 반드시 단조 증가해야 함
        timestamp_ms = int((time.perf_counter() - start_wall) * 1000)

        # 추론 시간 측정
        infer_start = time.perf_counter()
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        infer_end = time.perf_counter()

        latency_ms = (infer_end - infer_start) * 1000
        latency_history.append(latency_ms)
        if len(latency_history) > 30:
            latency_history.pop(0)
        avg_latency = float(np.mean(latency_history))

        # FPS 계산 (이동 평균)
        current_time = time.perf_counter()
        if prev_time != 0.0:
            curr_fps = 1.0 / max(current_time - prev_time, 1e-9)
            fps_history.append(curr_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = float(np.mean(fps_history))
        else:
            avg_fps = 0.0
        prev_time = current_time

        # ── 4. 랜드마크 시각화 ──────────────────────────────────────
        if result.pose_landmarks:
            draw_pose_landmarks(frame, result.pose_landmarks[0])

        # ── 5. 성능 지표 HUD 오버레이 ───────────────────────────────
        draw_metrics_overlay(frame, avg_fps, latency_ms, avg_latency)

        # ── 6. 화면 출력 ────────────────────────────────────────────
        cv2.imshow("Tro-Fit Vision AI — MediaPipe Tasks API Realtime Benchmark", frame)

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 사용자에 의해 벤치마크가 정지되었습니다.")
            break

    # ── 리소스 안전 해제 ─────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    if fps_history:
        print(f"\n[SUMMARY] 평균 FPS: {np.mean(fps_history):.1f} / 평균 Latency: {np.mean(latency_history):.1f} ms")
    print("[INFO] 웹캠 및 창 리소스가 해제되었습니다. 프로그램 종료.")


if __name__ == "__main__":
    run_mediapipe_webcam_benchmark()
