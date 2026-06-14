"""
snapshot_rom_pipeline.py — 타임스탬프 스냅샷 기반 ROM 분석 파이프라인
─────────────────────────────────────────────────────────────────────────────

설계 결정: 왜 IMAGE 모드인가?
  - VIDEO 모드는 프레임을 순차 처리하며 프레임 간 추적(tracking)을 활성화함
  - 스냅샷 ROM 측정은 3개의 독립적인 정지 이미지를 분석하는 작업
  - 각 프레임을 독립적으로 분석해야 하므로 → IMAGE 모드가 올바른 선택
  - 비디오 전체(96프레임)를 처리하지 않고 3프레임만 → 연산 효율 극대화

처리 흐름:
  1. 비디오 메타데이터 추출 (fps, 총 프레임 수, 해상도)
  2. 타임스탬프 → 프레임 번호 변환 (cap.set(POS_FRAMES) + cap.read())
  3. IMAGE 모드 PoseLandmarker로 각 프레임 독립 분석
  4. 최적 최대 자세 프레임 선택 (visibility 우선 → 각도 극단성)
  5. ROM = |neutral_angle − max_angle| per joint
  6. 결과 저장 (원본 이미지, 각도 시각화, landmark JSON, ROM JSON)

출력 구조:
  results/<video_stem>/
  ├── snapshots/
  │   ├── frames/
  │   │   ├── neutral.jpg              ← 중립 자세 원본
  │   │   ├── neutral_annotated.jpg    ← 중립 자세 + 랜드마크 오버레이
  │   │   ├── max_candidate_1.jpg      ← 후보 1 원본
  │   │   ├── max_candidate_2.jpg      ← 후보 2 원본
  │   │   ├── max_selected.jpg         ← 선택된 최대 자세 원본
  │   │   └── max_selected_annotated.jpg
  │   └── angle_vis/
  │       ├── neutral_angle.png        ← 중립 자세 관절 각도 시각화
  │       └── max_selected_angle.png   ← 최대 자세 관절 각도 시각화
  ├── landmark_json/
  │   ├── neutral.json
  │   ├── max_candidate_1.json
  │   ├── max_candidate_2.json
  │   └── max_selected.json
  └── rom_analysis_result.json         ← ★ 최종 ROM 결과

의존성:
  from core.landmark_extractor import create_image_landmarker, analyze_single_frame
  from core.angle_engine        import analyze_pose, VISIBILITY_THRESHOLD
  from core.visualizer          import build_angle_canvas
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── sys.path 설정 ──────────────────────────────────────────────────────────────
_PIPELINE_DIR     = Path(__file__).resolve().parent
_PIPELINE_PKG_DIR = _PIPELINE_DIR.parent           # mediapipe_rom_webcam_pipeline/
_PROJECT_ROOT_P   = (_PIPELINE_PKG_DIR / ".." / ".." / "..").resolve()

for _p in [str(_PIPELINE_PKG_DIR), str(_PROJECT_ROOT_P)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.vision_ai.mediapipe_rom_webcam_pipeline.core.angle_engine import (
        VISIBILITY_THRESHOLD, PoseAngleReport, analyze_pose,
    )
    from src.vision_ai.mediapipe_rom_webcam_pipeline.core.landmark_extractor import (
        analyze_single_frame, create_image_landmarker, draw_landmarks_on_frame,
    )
    from src.vision_ai.mediapipe_rom_webcam_pipeline.core.visualizer import build_angle_canvas
except ImportError:
    from core.angle_engine import VISIBILITY_THRESHOLD, PoseAngleReport, analyze_pose  # type: ignore
    from core.landmark_extractor import (  # type: ignore
        analyze_single_frame, create_image_landmarker, draw_landmarks_on_frame,
    )
    from core.visualizer import build_angle_canvas  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 컨테이너
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SnapshotFrame:
    """
    단일 스냅샷 프레임의 분석 결과를 담는 컨테이너.

    label       : "neutral", "max_candidate_1", "max_candidate_2" 등
    timestamp_s : 비디오에서의 실제 캡처 타임스탬프 (초)
    frame_index : 캡처된 실제 프레임 번호 (int(ts * fps))
    frame_bgr   : 원본 BGR 이미지 (np.ndarray)
    frame_record: extract_frame_record()가 생성한 Dict 스키마
    raw_result  : draw_landmarks_on_frame() 등 시각화에 필요한 원본 결과
    report      : analyze_pose()가 반환한 PoseAngleReport (감지 실패 시 None)
    """
    label:        str
    timestamp_s:  float
    frame_index:  int
    frame_bgr:    np.ndarray
    frame_record: dict
    raw_result:   Any              # vision.PoseLandmarkerResult (타입 힌트 순환 임포트 방지)
    report:       PoseAngleReport | None = None


# ──────────────────────────────────────────────────────────────────────────────
# 비디오 유틸리티
# ──────────────────────────────────────────────────────────────────────────────
def get_video_info(video_path: Path) -> dict:
    """
    비디오 메타데이터를 추출합니다.

    Returns:
        dict: {fps, total_frames, width, height, duration_s}

    Raises:
        RuntimeError: 비디오를 열 수 없을 때
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] 비디오를 열 수 없습니다: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration_s = total_frames / fps
    return {
        "fps":          round(fps, 3),
        "total_frames": total_frames,
        "width":        width,
        "height":       height,
        "duration_s":   round(duration_s, 3),
    }


def extract_frame_at_ts(
    video_path:  Path,
    timestamp_s: float,
    fps:         float,
) -> tuple[np.ndarray, int]:
    """
    특정 타임스탬프의 프레임을 직접 추출합니다.

    cap.set(CAP_PROP_POS_FRAMES, n)으로 O(1) 랜덤 접근 → 전체 디코딩 불필요.

    Args:
        video_path:  비디오 파일 경로
        timestamp_s: 캡처할 타임스탬프 (초)
        fps:         비디오 FPS

    Returns:
        tuple[np.ndarray, int]: (BGR 이미지, 실제 프레임 번호)

    Raises:
        RuntimeError: 프레임 읽기 실패 시
    """
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = int(round(timestamp_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(
            f"[ERROR] 프레임 추출 실패: {timestamp_s:.3f}s (frame #{frame_idx})\n"
            f"  총 프레임 수를 초과했거나 파일이 손상됐을 수 있습니다."
        )
    return frame_bgr, frame_idx


# ──────────────────────────────────────────────────────────────────────────────
# 최적 최대 자세 프레임 선택
# ──────────────────────────────────────────────────────────────────────────────
def _visibility_score(snap: SnapshotFrame) -> float:
    """포즈 감지 신뢰도 점수 (신뢰 관절 visibility 합산)."""
    if not snap.report:
        return 0.0
    return sum(
        min(j.visibility_a, j.visibility_v, j.visibility_c)
        for j in snap.report.reliable_joints()
    )


def _extremeness_score(snap: SnapshotFrame) -> float:
    """
    자세 극단성 점수 (더 극단적인 자세일수록 높음).

    우리 파이프라인의 모든 관절에서 각도가 '줄어들수록' 더 극단적 자세:
      - 어깨 외전: 중립 ~175° → 최대 ~50° (작을수록 더 든 상태)
      - 무릎 굴곡: 중립 ~180° → 최대 ~30°
    따라서 각도 역수의 합 = 극단성 지표.
    """
    if not snap.report:
        return 0.0
    angles = [
        j.angle_deg for j in snap.report.reliable_joints()
        if j.angle_deg is not None and j.angle_deg > 0
    ]
    if not angles:
        return 0.0
    return sum(1.0 / a for a in angles)


def select_best_max_frame(
    candidates: list[SnapshotFrame],
) -> tuple[SnapshotFrame, str]:
    """
    최대 자세 후보 중 ROM 측정에 가장 적합한 프레임을 선택합니다.

    선택 기준 (우선순위):
      1. 포즈 감지 여부 — 감지된 것 우선
      2. 관련 관절 visibility 합산 — 높을수록 신뢰도 ↑
         차이가 0.5 이상이면 visibility가 압도적으로 높은 것 선택
      3. 자세 극단성 점수 — 더 극단적(더 많이 들어올린) 자세 선택

    Returns:
        (선택된 SnapshotFrame, 선택 이유 문자열)
    """
    if len(candidates) == 1:
        return candidates[0], "only_one_candidate"

    # 포즈 감지된 후보만 필터
    detected = [c for c in candidates if c.report is not None]
    if not detected:
        raise RuntimeError("[ERROR] 모든 최대 자세 후보에서 포즈 감지 실패.")
    if len(detected) == 1:
        return detected[0], "other_candidate_no_detection"

    vis_scores    = [_visibility_score(c) for c in detected]
    vis_diff      = abs(vis_scores[0] - vis_scores[1])

    if vis_diff > 0.5:
        best_idx = int(np.argmax(vis_scores))
        return detected[best_idx], "higher_visibility"

    extreme_scores = [_extremeness_score(c) for c in detected]
    best_idx = int(np.argmax(extreme_scores))
    return detected[best_idx], "more_extreme_pose"


# ──────────────────────────────────────────────────────────────────────────────
# ROM 계산
# ──────────────────────────────────────────────────────────────────────────────
def compute_snapshot_rom(
    neutral:      SnapshotFrame,
    max_selected: SnapshotFrame,
) -> dict:
    """
    중립 자세 + 최대 자세 스냅샷으로부터 관절별 ROM을 계산합니다.

    ROM = |neutral_angle − max_angle|

    어깨 외전 예시:
      neutral  ~175.3°  (팔 붙인 자세, Elbow→Shoulder→Hip 벡터 사이각)
      max      ~62.7°   (팔 최대 외전, 각도 감소)
      ROM  = |175.3 − 62.7| = 112.6°

    신뢰도 분류:
      HIGH   : 두 프레임 모두 reliable, 각도 차이 ≤ 3° (선택 일관성 확인용)
      MEDIUM : reliable이지만 후보 간 각도 차이 3~7°
      LOW    : 어느 한 쪽이 reliable하지 않음

    Returns:
        dict: 관절별 ROM 딕셔너리
    """
    if not neutral.report or not max_selected.report:
        return {}

    neutral_angles: dict[str, float] = {
        j.joint: j.angle_deg
        for j in neutral.report.joints
        if j.reliable and j.angle_deg is not None
    }
    max_angles: dict[str, float] = {
        j.joint: j.angle_deg
        for j in max_selected.report.joints
        if j.reliable and j.angle_deg is not None
    }

    rom_results: dict[str, dict] = {}
    all_joints = set(neutral_angles) | set(max_angles)

    for joint in sorted(all_joints):
        n_ang = neutral_angles.get(joint)
        m_ang = max_angles.get(joint)

        if n_ang is not None and m_ang is not None:
            rom_val = round(abs(n_ang - m_ang), 2)
            rom_results[joint] = {
                "neutral_angle": round(n_ang, 2),
                "max_angle":     round(m_ang, 2),
                "rom":           rom_val,
                "reliable":      True,
            }
        else:
            rom_results[joint] = {
                "neutral_angle": round(n_ang, 2) if n_ang else None,
                "max_angle":     round(m_ang, 2) if m_ang else None,
                "rom":           None,
                "reliable":      False,
                "reason": "neutral_not_reliable" if n_ang is None else "max_not_reliable",
            }

    return rom_results


def _overall_confidence(
    candidates:   list[SnapshotFrame],
    selected:     SnapshotFrame,
    rom_results:  dict,
) -> str:
    """
    측정 전체 신뢰도를 평가합니다.

    HIGH   : 신뢰 관절 5개 이상 + 후보 프레임 간 일관성 양호
    MEDIUM : 신뢰 관절 2~4개 또는 후보 간 각도 차이 다소 있음
    LOW    : 신뢰 관절 1개 이하 또는 포즈 감지 불안정
    """
    reliable_count = sum(1 for v in rom_results.values() if v.get("reliable"))

    if reliable_count >= 5:
        return "HIGH"
    elif reliable_count >= 2:
        return "MEDIUM"
    return "LOW"


# ──────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인 실행
# ──────────────────────────────────────────────────────────────────────────────
def run_snapshot_rom_pipeline(
    video_path:   str | Path,
    model_path:   str | Path,
    output_dir:   str | Path,
    neutral_ts:   float = 0.5,
    max_ts_list:  list[float] | None = None,
    use_world:    bool  = True,
    threshold:    float = VISIBILITY_THRESHOLD,
    save_images:  bool  = True,
) -> dict:
    """
    특정 타임스탬프의 스냅샷 3장으로 ROM을 분석합니다.

    Args:
        video_path:   입력 비디오 파일 경로
        model_path:   pose_landmarker_full.task 파일 경로
        output_dir:   결과 저장 루트 디렉토리
        neutral_ts:   중립 자세 캡처 타임스탬프 (초, 기본값: 0.5)
        max_ts_list:  최대 자세 후보 타임스탬프 목록 (None = 자동: 끝-1.0s, 끝-0.5s)
        use_world:    True → world_landmarks 기반 각도 계산 (권장)
        threshold:    visibility 임계값 (기본: 0.65)
        save_images:  True → 원본 + 각도 시각화 이미지 저장

    Returns:
        dict: 전체 ROM 분석 결과 (rom_analysis_result.json과 동일)

    Raises:
        FileNotFoundError: 비디오 또는 모델 파일 없음
        RuntimeError:      포즈 감지 실패 또는 프레임 추출 실패
    """
    video_path = Path(video_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"[ERROR] 비디오 파일을 찾을 수 없습니다: {video_path}")

    # ── 비디오 메타데이터 ──────────────────────────────────────────────
    info      = get_video_info(video_path)
    fps       = info["fps"]
    duration  = info["duration_s"]
    width     = info["width"]
    height    = info["height"]

    # ── 타임스탬프 결정 ────────────────────────────────────────────────
    # max_ts_list=None → 영상 끝-1.0s 와 끝-0.5s 자동 계산
    if max_ts_list is None:
        max_ts_list = [
            round(max(neutral_ts + 0.5, duration - 1.0), 3),
            round(max(neutral_ts + 1.0, duration - 0.5), 3),
        ]

    # 범위 보호: 0 ≤ ts < duration
    clamp = lambda t: max(0.0, min(t, duration - 1.0 / fps))
    neutral_ts  = clamp(neutral_ts)
    max_ts_list = [clamp(t) for t in max_ts_list]

    start_wall = time.perf_counter()

    print(f"\n{'='*64}")
    print(f"  [Tro-Fit] Snapshot ROM Pipeline")
    print(f"{'='*64}")
    print(f"  비디오    : {video_path.name}")
    print(f"  해상도    : {width}x{height}  /  {fps} FPS  /  {duration:.2f}초")
    print(f"  중립 캡처 : {neutral_ts:.3f}s  → frame #{int(round(neutral_ts * fps))}")
    for i, ts in enumerate(max_ts_list, 1):
        print(f"  최대 후보{i}: {ts:.3f}s  → frame #{int(round(ts * fps))}")
    print(f"  world 좌표: {'사용 (미터 단위, 권장)' if use_world else '미사용 (정규화 좌표)'}")
    print(f"  threshold : {threshold}")
    print(f"{'='*64}\n")

    # ── 출력 디렉토리 구성 ─────────────────────────────────────────────
    frames_dir    = output_dir / "snapshots" / "frames"
    angle_vis_dir = output_dir / "snapshots" / "angle_vis"
    lm_json_dir   = output_dir / "landmark_json"

    lm_json_dir.mkdir(parents=True, exist_ok=True)
    if save_images:
        frames_dir.mkdir(parents=True, exist_ok=True)
        angle_vis_dir.mkdir(parents=True, exist_ok=True)

    # ── IMAGE 모드 PoseLandmarker 초기화 ──────────────────────────────
    # min_pose_detection_confidence=0.3: 스냅샷은 정지 이미지이므로 추적 없이
    # 단일 추론으로 검출해야 함. 낮은 임계값으로 최대한 감지하고,
    # 실제 각도 계산 시 visibility 게이트(threshold=0.65)로 신뢰도 보장.
    snapshots: list[SnapshotFrame] = []

    print("[INFO] IMAGE 모드 PoseLandmarker 초기화...")
    with create_image_landmarker(model_path) as landmarker:

        # ── 1. 중립 자세 프레임 처리 ──────────────────────────────────
        print(f"\n[STEP 1] 중립 자세 캡처: {neutral_ts:.3f}s")
        neutral_bgr, neutral_fi = extract_frame_at_ts(video_path, neutral_ts, fps)
        neutral_record, neutral_raw = analyze_single_frame(
            landmarker, neutral_bgr,
            frame_index=neutral_fi,
            timestamp_ms=int(neutral_ts * 1000),
        )

        neutral_report = None
        if neutral_record["detected"] and neutral_record["poses"]:
            neutral_report = analyze_pose(
                neutral_record["poses"][0],
                frame_index=neutral_fi,
                timestamp_ms=int(neutral_ts * 1000),
                threshold=threshold,
                use_world=use_world,
            )

        neutral_snap = SnapshotFrame(
            label="neutral",
            timestamp_s=neutral_ts,
            frame_index=neutral_fi,
            frame_bgr=neutral_bgr,
            frame_record=neutral_record,
            raw_result=neutral_raw,
            report=neutral_report,
        )

        status = "✅ 감지됨" if neutral_record["detected"] else "❌ 감지 실패"
        print(f"  결과: {status} ({neutral_record['num_poses']} pose(s))")
        if neutral_report:
            for j in neutral_report.reliable_joints():
                print(f"    {j.joint:22s}: {j.angle_deg:7.2f}°")

        if not neutral_record["detected"]:
            raise RuntimeError(
                "[ERROR] 중립 자세 프레임에서 포즈가 감지되지 않았습니다.\n"
                f"  → {neutral_ts:.3f}s 주변에서 전신이 프레임에 들어오는지 확인하세요."
            )

        # landmark JSON 저장
        (lm_json_dir / "neutral.json").write_text(
            json.dumps(neutral_record, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        # ── 2. 최대 자세 후보 프레임들 처리 ───────────────────────────
        print(f"\n[STEP 2] 최대 자세 후보 캡처: {len(max_ts_list)}개")
        max_candidates: list[SnapshotFrame] = []

        for i, ts in enumerate(max_ts_list, 1):
            label = f"max_candidate_{i}"
            print(f"\n  [{i}] {ts:.3f}s (label: {label})")

            frame_bgr, frame_fi = extract_frame_at_ts(video_path, ts, fps)
            frame_record, frame_raw = analyze_single_frame(
                landmarker, frame_bgr,
                frame_index=frame_fi,
                timestamp_ms=int(ts * 1000),
            )

            report = None
            if frame_record["detected"] and frame_record["poses"]:
                report = analyze_pose(
                    frame_record["poses"][0],
                    frame_index=frame_fi,
                    timestamp_ms=int(ts * 1000),
                    threshold=threshold,
                    use_world=use_world,
                )

            snap = SnapshotFrame(
                label=label,
                timestamp_s=ts,
                frame_index=frame_fi,
                frame_bgr=frame_bgr,
                frame_record=frame_record,
                raw_result=frame_raw,
                report=report,
            )
            max_candidates.append(snap)

            status = "✅ 감지됨" if frame_record["detected"] else "❌ 감지 실패"
            print(f"    결과: {status} ({frame_record['num_poses']} pose(s))")
            if report:
                for j in report.reliable_joints():
                    print(f"      {j.joint:22s}: {j.angle_deg:7.2f}°")

            # landmark JSON 저장
            (lm_json_dir / f"{label}.json").write_text(
                json.dumps(frame_record, ensure_ascii=False, indent=2), encoding="utf-8",
            )

        # ── 3. 최적 최대 자세 선택 ────────────────────────────────────
        print(f"\n[STEP 3] 최적 최대 자세 선택")
        best_max, reason = select_best_max_frame(max_candidates)
        print(f"  → 선택: {best_max.label} ({best_max.timestamp_s:.3f}s) / 이유: {reason}")

        # 선택된 max landmark JSON 저장
        (lm_json_dir / "max_selected.json").write_text(
            json.dumps(best_max.frame_record, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        # ── 4. ROM 계산 ───────────────────────────────────────────────
        print(f"\n[STEP 4] ROM 계산")
        rom_results = compute_snapshot_rom(neutral_snap, best_max)
        confidence  = _overall_confidence(max_candidates, best_max, rom_results)

        print(f"  {'관절명':24s}  {'중립각':>8s}  {'최대각':>8s}  {'ROM':>8s}")
        print(f"  {'-'*56}")
        for joint, data in rom_results.items():
            if data["reliable"]:
                print(
                    f"  {joint:24s}  "
                    f"{data['neutral_angle']:7.2f}°  "
                    f"{data['max_angle']:7.2f}°  "
                    f"{data['rom']:7.2f}°"
                )
            else:
                reason_str = data.get("reason", "unknown")
                print(f"  {joint:24s}  {'—':>8s}  {'—':>8s}  [UNRELIABLE: {reason_str}]")

        # ── 5. 이미지 저장 ────────────────────────────────────────────
        if save_images:
            print(f"\n[STEP 5] 이미지 저장")

            # 원본 프레임 저장
            cv2.imwrite(str(frames_dir / "neutral.jpg"), neutral_snap.frame_bgr)
            cv2.imwrite(str(frames_dir / "max_selected.jpg"), best_max.frame_bgr)
            for snap in max_candidates:
                cv2.imwrite(str(frames_dir / f"{snap.label}.jpg"), snap.frame_bgr)

            # 랜드마크 오버레이 이미지 저장
            neutral_ann = draw_landmarks_on_frame(neutral_snap.frame_bgr, neutral_snap.raw_result)
            cv2.imwrite(str(frames_dir / "neutral_annotated.jpg"), neutral_ann)

            max_ann = draw_landmarks_on_frame(best_max.frame_bgr, best_max.raw_result)
            cv2.imwrite(str(frames_dir / "max_selected_annotated.jpg"), max_ann)

            # 각도 시각화 캔버스 저장
            if neutral_report and neutral_record["poses"]:
                neutral_canvas = build_angle_canvas(neutral_record["poses"][0], neutral_report)
                cv2.imwrite(str(angle_vis_dir / "neutral_angle.png"), neutral_canvas)

            if best_max.report and best_max.frame_record["poses"]:
                max_canvas = build_angle_canvas(
                    best_max.frame_record["poses"][0], best_max.report,
                )
                cv2.imwrite(str(angle_vis_dir / "max_selected_angle.png"), max_canvas)

            print(f"  저장 위치: {frames_dir}")

    # ── 최종 결과 구성 + JSON 저장 ────────────────────────────────────
    elapsed = time.perf_counter() - start_wall

    final_result = {
        "video_file":   video_path.name,
        "video_info":   info,
        "measurement": {
            "neutral_captured_at":  f"{neutral_ts:.3f}s (frame #{neutral_snap.frame_index})",
            "max_candidates":       [
                f"{c.timestamp_s:.3f}s (frame #{c.frame_index})"
                for c in max_candidates
            ],
            "max_selected":         f"{best_max.timestamp_s:.3f}s (frame #{best_max.frame_index})",
            "selection_reason":     reason,
            "use_world_landmarks":  use_world,
            "visibility_threshold": threshold,
        },
        "rom_results":   rom_results,
        "confidence":    confidence,
        "elapsed_sec":   round(elapsed, 3),
        "model":         "pose_landmarker_full.task",
    }

    rom_path = output_dir / "rom_analysis_result.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    rom_path.write_text(
        json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"\n{'='*64}")
    print(f"  [완료] Snapshot ROM Pipeline")
    print(f"  신뢰도   : {confidence}")
    print(f"  처리 시간: {elapsed:.2f}초")
    print(f"  결과 저장: {rom_path}")
    print(f"{'='*64}\n")

    return final_result
