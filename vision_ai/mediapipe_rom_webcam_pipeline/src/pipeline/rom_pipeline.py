"""
rom_pipeline.py — ROM 분석 파이프라인 핵심 실행 엔진
─────────────────────────────────────────────────────────────────────────────

책임:
  - 비디오 파일 처리 루프 (run_video_pipeline)
  - 프레임별 랜드마크 추출 → ROM 분석 → 시각화 → 결과 저장
  - 최대/최소 ROM 자동 추적 (ROMTracker)
  - FPS / Latency 성능 측정
  - 결과 JSON 저장 (프레임별 + 통합)

설계 포인트:
  - VIDEO 모드 + 타임스탬프 단조증가 → 프레임 간 추적 최적화
  - use_world=True 옵션 → world_landmarks 기반 각도 계산 (더 정확)
  - max_frames 옵션 → 부분 처리 및 테스트 지원
  - preview 옵션 → 실시간 프리뷰 창 (q키 종료)
  - context manager로 PoseLandmarker 자원 안전 해제

의존성:
  from core.landmark_extractor import create_landmarker, extract_frame_record, draw_landmarks_on_frame
  from core.angle_engine import analyze_pose
  from core.visualizer import render_angle_images, generate_pictographic_svg
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# sys.path 설정 (이 파일 위치: vision_ai/mediapipe_rom_webcam_pipeline/src/pipeline/rom_pipeline.py)
_PIPELINE_DIR = Path(__file__).resolve().parent          # .../pipeline/
_PIPELINE_PKG_DIR = _PIPELINE_DIR.parent                 # .../src/
_SRC_PKG_DIR  = _PIPELINE_PKG_DIR.parent                 # .../mediapipe_rom_webcam_pipeline/
_PROJECT_ROOT_P   = (_PIPELINE_PKG_DIR / ".." / ".." / "..").resolve()  # trofit/

for _p in [str(_PIPELINE_PKG_DIR), str(_PROJECT_ROOT_P)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from vision_ai.mediapipe_rom_webcam_pipeline.src.core.angle_engine import (
        VISIBILITY_THRESHOLD, PoseAngleReport, analyze_pose,
    )
    from vision_ai.mediapipe_rom_webcam_pipeline.src.core.landmark_extractor import (
        create_landmarker, draw_landmarks_on_frame, draw_performance_hud, extract_frame_record,
    )
    from vision_ai.mediapipe_rom_webcam_pipeline.src.core.visualizer import (
        generate_pictographic_svg, render_angle_images,
    )
except ImportError:
    from core.angle_engine import (  # type: ignore
        VISIBILITY_THRESHOLD, PoseAngleReport, analyze_pose,
    )
    from core.landmark_extractor import (  # type: ignore
        create_landmarker, draw_landmarks_on_frame, draw_performance_hud, extract_frame_record,
    )
    from core.visualizer import generate_pictographic_svg, render_angle_images  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# ROM 최대/최소 추적기
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ROMTracker:
    """
    동작 구간 전체에서 관절별 최대/최소 ROM을 추적합니다.

    피트니스 앱에서 "스쿼트 최대 무릎 굴곡각" 같은 값을 추출할 때 사용합니다.
    """
    max_angles: dict[str, float] = field(default_factory=dict)
    min_angles: dict[str, float] = field(default_factory=dict)
    rom:        dict[str, float] = field(default_factory=dict)  # max - min

    def update(self, reports: list[PoseAngleReport]) -> None:
        """신뢰할 수 있는 관절 각도로 max/min을 갱신합니다."""
        for report in reports:
            for joint in report.reliable_joints():
                name = joint.joint
                deg  = joint.angle_deg
                if deg is None:
                    continue
                if name not in self.max_angles or deg > self.max_angles[name]:
                    self.max_angles[name] = deg
                if name not in self.min_angles or deg < self.min_angles[name]:
                    self.min_angles[name] = deg

    def compute_rom(self) -> dict[str, float]:
        """ROM = max - min을 계산하여 반환합니다."""
        self.rom = {
            name: round(self.max_angles[name] - self.min_angles.get(name, self.max_angles[name]), 2)
            for name in self.max_angles
        }
        return self.rom

    def to_dict(self) -> dict:
        return {
            "max_angles": {k: round(v, 2) for k, v in self.max_angles.items()},
            "min_angles": {k: round(v, 2) for k, v in self.min_angles.items()},
            "rom":        self.compute_rom(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인 실행
# ──────────────────────────────────────────────────────────────────────────────
def run_video_pipeline(
    video_path:   str | Path,
    model_path:   str | Path,
    output_dir:   str | Path,
    use_world:    bool  = True,
    preview:      bool  = False,
    max_frames:   int   = 0,
    threshold:    float = VISIBILITY_THRESHOLD,
    save_angle_images:    bool = True,
    save_pictographic:    bool = True,
) -> dict:
    """
    비디오 파일에 대한 전체 ROM 분석 파이프라인을 실행합니다.

    처리 흐름:
      1. VideoCapture → 프레임별 읽기
      2. BGR → RGB → mp.Image 변환
      3. detect_for_video() (VIDEO 모드, 타임스탬프 단조증가)
      4. extract_frame_record() → Dict 스키마 생성
         (normalized landmarks + world_landmarks + timestamp_ms)
      5. analyze_pose()  → PoseAngleReport (visibility 게이트 포함)
      6. ROMTracker 갱신 (프레임별 max/min 추적)
      7. 시각화 이미지 / SVG 픽토그래픽 저장 (선택)
      8. 결과 JSON 저장

    출력 구조:
      output_dir/
      ├── landmark_json/
      │   ├── frame_0000.json      ← 프레임별 랜드마크 + world_landmarks
      │   ├── frame_0001.json
      │   └── landmarks_all.json   ← 전체 통합 JSON
      ├── angle_json/
      │   ├── frame_0000_angle.json
      │   └── angle_all.json
      ├── angle_img/               ← 각도 시각화 이미지 (선택)
      │   └── frame_0000_angle.png
      └── pictographic/            ← SVG 픽토그래픽 (선택)
          └── frame_0000.svg

    Args:
        video_path:         입력 비디오 파일 경로 (MP4, AVI, MOV 등 OpenCV 지원 포맷)
        model_path:         pose_landmarker_full.task 파일 경로
        output_dir:         결과 저장 루트 디렉토리
        use_world:          True → world_landmarks 기반 각도 계산 (권장)
        preview:            True → 실시간 프리뷰 창 표시 (q키 종료)
        max_frames:         0 → 전체 처리 / N → 최대 N 프레임만 처리
        threshold:          visibility 임계값 (기본: 0.65)
        save_angle_images:  True → 각도 시각화 이미지 저장
        save_pictographic:  True → SVG 픽토그래픽 저장

    Returns:
        dict: {
            "summary": { ... },         ← 전체 처리 요약
            "rom_tracker": { ... },     ← 관절별 ROM 결과
            "output_dir": str,
        }
    """
    video_path = Path(video_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"[ERROR] 비디오 파일을 찾을 수 없습니다: {video_path}")

    # ── 출력 디렉토리 구성 ─────────────────────────────────────────────
    lm_json_dir    = output_dir / "landmark_json"
    angle_json_dir = output_dir / "angle_json"
    angle_img_dir  = output_dir / "angle_img"
    picto_dir      = output_dir / "pictographic"

    lm_json_dir.mkdir(parents=True, exist_ok=True)
    angle_json_dir.mkdir(parents=True, exist_ok=True)
    if save_angle_images:
        angle_img_dir.mkdir(parents=True, exist_ok=True)
    if save_pictographic:
        picto_dir.mkdir(parents=True, exist_ok=True)

    # ── 비디오 캡처 초기화 ─────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] 비디오 파일을 열 수 없습니다: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 1:
        video_fps = 30.0  # FPS 정보 없는 비디오 대응

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n[Tro-Fit] ROM Pipeline 시작")
    print(f"  입력 비디오 : {video_path.name}")
    print(f"  해상도      : {width}x{height} @ {video_fps:.1f} FPS")
    print(f"  전체 프레임 : {total_frames}")
    print(f"  world 좌표  : {'사용 (더 정확)' if use_world else '미사용 (정규화 좌표)'}")
    print(f"  출력 디렉토리: {output_dir}\n")

    # ── 성능 측정 변수 ─────────────────────────────────────────────────
    latency_history: list[float] = []
    fps_history:     list[float] = []
    prev_time: float = 0.0
    start_wall = time.perf_counter()

    # ── 결과 축적 ──────────────────────────────────────────────────────
    all_landmarks:    dict[str, dict] = {}
    all_angle_results: dict[str, dict] = {}
    rom_tracker = ROMTracker()

    processed_frames = 0
    detected_frames  = 0

    # ── 파이프라인 실행 ────────────────────────────────────────────────
    with create_landmarker(model_path) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if max_frames > 0 and frame_idx >= max_frames:
                break

            # VIDEO 모드 타임스탬프 (단조증가 필수)
            timestamp_ms = int(frame_idx * (1000.0 / video_fps))

            # BGR → RGB → mp.Image
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # ── 추론 ───────────────────────────────────────────────────
            infer_start = time.perf_counter()
            result      = landmarker.detect_for_video(mp_image, timestamp_ms)
            infer_end   = time.perf_counter()

            latency_ms = (infer_end - infer_start) * 1000
            latency_history.append(latency_ms)
            if len(latency_history) > 30:
                latency_history.pop(0)
            avg_latency = float(np.mean(latency_history))

            # FPS 계산
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

            # ── 랜드마크 추출 ──────────────────────────────────────────
            frame_record = extract_frame_record(
                result, frame_idx, timestamp_ms, width, height,
            )
            frame_key = f"frame_{frame_idx:06d}"

            all_landmarks[frame_key] = frame_record
            if frame_record["detected"]:
                detected_frames += 1

            # 프레임별 landmark JSON 저장
            lm_path = lm_json_dir / f"{frame_key}.json"
            lm_path.write_text(
                json.dumps(frame_record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # ── ROM 분석 ───────────────────────────────────────────────
            reports: list[PoseAngleReport] = [
                analyze_pose(
                    pose,
                    frame_index=frame_idx,
                    timestamp_ms=timestamp_ms,
                    threshold=threshold,
                    use_world=use_world,
                )
                for pose in frame_record["poses"]
            ]
            rom_tracker.update(reports)

            angle_data = {
                "frame_index":  frame_idx,
                "timestamp_ms": timestamp_ms,
                "num_poses":    frame_record["num_poses"],
                "poses":        [r.to_dict() for r in reports],
            }
            all_angle_results[frame_key] = angle_data

            angle_json_path = angle_json_dir / f"{frame_key}_angle.json"
            angle_json_path.write_text(
                json.dumps(angle_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # ── 시각화 저장 ────────────────────────────────────────────
            if save_angle_images and reports:
                render_angle_images(
                    name=frame_key,
                    raw_data=frame_record,
                    reports=reports,
                    out_dir=angle_img_dir,
                )

            if save_pictographic and frame_record["detected"]:
                svg_path = picto_dir / f"{frame_key}.svg"
                generate_pictographic_svg(
                    poses_data=frame_record["poses"],
                    image_width=width,
                    image_height=height,
                    output_path=svg_path,
                )

            # ── 프리뷰 ─────────────────────────────────────────────────
            if preview:
                preview_frame = draw_landmarks_on_frame(frame_bgr, result)
                draw_performance_hud(preview_frame, avg_fps, latency_ms, avg_latency)
                cv2.imshow("Tro-Fit ROM Pipeline — Preview (q to quit)", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] 사용자가 프리뷰를 종료했습니다.")
                    break

            processed_frames += 1
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(
                    f"  [{frame_idx:5d}/{total_frames}]  "
                    f"FPS: {avg_fps:5.1f}  Latency: {avg_latency:5.1f}ms  "
                    f"Detected: {frame_record['num_poses']} pose(s)"
                )

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    # ── 통합 JSON 저장 ─────────────────────────────────────────────────
    all_lm_path = lm_json_dir / "landmarks_all.json"
    all_lm_path.write_text(
        json.dumps(all_landmarks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_angle_path = angle_json_dir / "angle_all.json"
    all_angle_path.write_text(
        json.dumps(all_angle_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ROM 요약 저장
    rom_result = rom_tracker.to_dict()
    rom_path   = output_dir / "rom_analysis_result.json"
    rom_path.write_text(
        json.dumps(rom_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── 최종 요약 ──────────────────────────────────────────────────────
    elapsed     = time.perf_counter() - start_wall
    final_avg_fps = float(np.mean(fps_history)) if fps_history else 0.0
    final_avg_lat = float(np.mean(latency_history)) if latency_history else 0.0

    summary = {
        "video_file":       video_path.name,
        "video_fps":        round(video_fps, 2),
        "resolution":       f"{width}x{height}",
        "processed_frames": processed_frames,
        "detected_frames":  detected_frames,
        "detection_rate":   round(detected_frames / max(processed_frames, 1) * 100, 1),
        "elapsed_sec":      round(elapsed, 2),
        "avg_fps":          round(final_avg_fps, 2),
        "avg_latency_ms":   round(final_avg_lat, 2),
        "use_world_landmarks": use_world,
        "visibility_threshold": threshold,
    }

    print(f"\n{'='*60}")
    print(f"  [ROM Pipeline 완료]")
    print(f"  처리 프레임 : {processed_frames} / {total_frames}")
    print(f"  감지율      : {summary['detection_rate']}%")
    print(f"  평균 FPS    : {final_avg_fps:.1f}")
    print(f"  평균 레이턴시: {final_avg_lat:.1f} ms")
    print(f"  처리 시간   : {elapsed:.1f} 초")
    print(f"\n  ROM 결과 (전체 동작 구간):")
    for joint, rom_val in rom_result.get("rom", {}).items():
        max_v = rom_result["max_angles"].get(joint, 0)
        min_v = rom_result["min_angles"].get(joint, 0)
        print(f"    {joint:25s}: ROM={rom_val:6.1f}°  (max={max_v:.1f}° / min={min_v:.1f}°)")
    print(f"\n  결과 저장 위치: {output_dir}")
    print(f"    ROM JSON  : {rom_path}")
    print(f"{'='*60}\n")

    return {
        "summary":    summary,
        "rom_tracker": rom_result,
        "output_dir": str(output_dir),
    }
