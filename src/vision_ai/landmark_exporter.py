"""
Export frame-by-frame 3D pose landmarks from a webcam or video file.

This script uses MediaPipe Tasks PoseLandmarker and writes a JSON file with:
- normalized image landmarks: x, y, z, visibility, presence
- world landmarks: x, y, z, visibility, presence

Examples:
    python src/vision_ai/landmark_exporter.py --video sample.mp4 --output results/landmarks.json
    python src/vision_ai/landmark_exporter.py --webcam --output results/webcam_landmarks.json --max-frames 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.vision_ai.rom_prototype.angle_calculator_test_plus import build_diagnosis_payload
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pose_landmarker_full.task")


LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (27, 31), (28, 32),
]


def _optional_float(obj: Any, attr: str) -> float | None:
    value = getattr(obj, attr, None)
    if value is None:
        return None
    return float(value)


def landmark_to_dict(landmark: Any, index: int) -> dict[str, Any]:
    """Convert a MediaPipe landmark object into a JSON-friendly dict."""
    return {
        "id": index,
        "name": LANDMARK_NAMES[index] if index < len(LANDMARK_NAMES) else f"landmark_{index}",
        "x": float(landmark.x),
        "y": float(landmark.y),
        "z": float(landmark.z),
        "visibility": _optional_float(landmark, "visibility"),
        "presence": _optional_float(landmark, "presence"),
    }


def create_landmarker(model_path: str) -> vision.PoseLandmarker:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Download it from the project root with:\n"
            "  New-Item -ItemType Directory -Force -Path models\n"
            "  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task "
            "-OutFile models/pose_landmarker_full.task"
        )

    with open(model_path, "rb") as model_file:
        model_buffer = model_file.read()

    base_options = python.BaseOptions(model_asset_buffer=model_buffer)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def draw_pose_overlay(frame_bgr: Any, landmarks: list[Any]) -> None:
    """Draw pose joints and skeleton lines on the preview frame."""
    if not landmarks:
        cv2.putText(
            frame_bgr,
            "NO POSE DETECTED",
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    height, width = frame_bgr.shape[:2]

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(frame_bgr, start_point, end_point, (255, 120, 0), 2)

    for index, landmark in enumerate(landmarks):
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame_bgr, point, 5, (0, 230, 0), -1)
        cv2.circle(frame_bgr, point, 5, (0, 80, 0), 1)
        if index in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
            cv2.putText(
                frame_bgr,
                str(index),
                (point[0] + 5, point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        frame_bgr,
        f"POSE DETECTED: {len(landmarks)} joints",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 230, 0),
        2,
        cv2.LINE_AA,
    )

def result_to_frame_record(
    result: vision.PoseLandmarkerResult,
    frame_index: int,
    timestamp_ms: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    pose_landmarks = result.pose_landmarks[0] if result.pose_landmarks else []
    world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else []

    return {
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "image_width": width,
        "image_height": height,
        "detected": bool(pose_landmarks),
        "landmark_count": len(pose_landmarks),
        "landmarks": [
            landmark_to_dict(landmark, index)
            for index, landmark in enumerate(pose_landmarks)
        ],
        "world_landmarks": [
            landmark_to_dict(landmark, index)
            for index, landmark in enumerate(world_landmarks)
        ],
    }


def export_landmarks(
    source: int | str,
    output_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    max_frames: int | None = None,
    preview: bool = False,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames: list[dict[str, Any]] = []
    frame_index = 0
    detected_frames = 0
    started_at = time.perf_counter()

    with create_landmarker(model_path) as landmarker:
        while cap.isOpened():
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_index * (1000.0 / fps))

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            record = result_to_frame_record(
                result=result,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                width=frame_bgr.shape[1],
                height=frame_bgr.shape[0],
            )
            frames.append(record)
            if record["detected"]:
                detected_frames += 1

            if preview:
                preview_frame = frame_bgr.copy()
                draw_pose_overlay(preview_frame, result.pose_landmarks[0] if result.pose_landmarks else [])
                cv2.imshow("Tro-Fit Landmark Exporter", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    elapsed_sec = time.perf_counter() - started_at
    detection_rate = (detected_frames / len(frames) * 100.0) if frames else 0.0

    payload = {
        "project_name": "Tro-Fit",
        "export_type": "pose_landmarks_3d",
        "model_path": os.path.relpath(model_path, PROJECT_ROOT),
        "source": str(source),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": len(frames),
        "detected_frames": detected_frames,
        "detection_rate_percent": round(detection_rate, 2),
        "elapsed_seconds": round(elapsed_sec, 3),
        "landmark_schema": {
            "landmarks": "Normalized image coordinates. x/y are usually 0..1; z is relative depth.",
            "world_landmarks": "MediaPipe world coordinates for 3D pose landmarks.",
            "count": 33,
        },
        "frames": frames,
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    return payload



def post_diagnosis_payload(api_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body) if response_body else {}
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Diagnosis API returned HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to diagnosis API: {exc.reason}") from exc
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MediaPipe 3D pose landmarks to JSON.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video", type=str, help="Path to a video file.")
    source_group.add_argument("--webcam", action="store_true", help="Use a webcam or mobile camera exposed as webcam.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index for --webcam.")
    parser.add_argument("--output", type=str, default="results/landmarks.json", help="Output JSON path.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to pose_landmarker .task model.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit.")
    parser.add_argument("--preview", action="store_true", help="Show a preview window. Press q to stop.")
    parser.add_argument("--user-id", type=str, default=None, help="User ID to save diagnosis data for.")
    parser.add_argument("--send-to-api", action="store_true", help="Send scan summary/ROM diagnosis to backend after export.")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000/api/v1/diagnosis/", help="Diagnosis API endpoint URL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source: int | str = args.camera_index if args.webcam else args.video

    result = export_landmarks(
        source=source,
        output_path=args.output,
        model_path=args.model,
        max_frames=args.max_frames,
        preview=args.preview,
    )

    print("[SUCCESS] Landmark JSON exported")
    print(f"  output          : {args.output}")
    print(f"  total frames    : {result['total_frames']}")
    print(f"  detected frames : {result['detected_frames']}")
    print(f"  detection rate  : {result['detection_rate_percent']}%")

    if args.send_to_api or args.user_id:
        if not args.user_id:
            raise ValueError("--user-id is required when sending diagnosis data to the backend.")
        diagnosis_payload = build_diagnosis_payload(result, args.user_id, args.output)
        api_response = post_diagnosis_payload(args.api_url, diagnosis_payload)
        print("[SUCCESS] Diagnosis saved to backend")
        print(f"  api url         : {args.api_url}")
        print(f"  session id      : {api_response.get('session_id')}")
        print(f"  falls score     : {api_response.get('diagnosis', {}).get('falls_risk_score')}")


if __name__ == "__main__":
    main()








