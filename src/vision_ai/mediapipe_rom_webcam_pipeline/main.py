"""
main.py — Tro-Fit Vision AI ROM 분석 파이프라인 진입점
─────────────────────────────────────────────────────────────────────────────

모드:
  snapshot (기본) — 특정 타임스탬프 3장 스냅샷으로 ROM 측정
                    중립 1장 + 최대 후보 2장 → 더 나은 것 선택 → ROM 계산
  full            — 비디오 전체 프레임 처리 (연속 ROM 추적)

사용법:
    # snapshot 모드 (기본): 4초 영상에서 0.5s 중립 + 끝-1s, 끝-0.5s 최대 캡처
    python main.py --video videos/shoulder_test.mp4

    # snapshot 모드 타임스탬프 직접 지정
    python main.py --video videos/shoulder_test.mp4 --neutral-ts 0.5 --max-ts 3.0,3.5

    # full 모드: 전체 프레임 순차 처리 + 실시간 프리뷰
    python main.py --video videos/squat.mp4 --mode full --preview

    # full 모드: 처음 100 프레임만 (테스트용)
    python main.py --video videos/test.mp4 --mode full --max-frames 100

    # 전체 옵션 확인
    python main.py --help

snapshot 출력 구조:
    results/<video_stem>/
    ├── snapshots/
    │   ├── frames/
    │   │   ├── neutral.jpg              ← 중립 자세 원본
    │   │   ├── neutral_annotated.jpg    ← 랜드마크 오버레이
    │   │   ├── max_candidate_1.jpg
    │   │   ├── max_candidate_2.jpg
    │   │   ├── max_selected.jpg         ← 선택된 최대 자세 원본
    │   │   └── max_selected_annotated.jpg
    │   └── angle_vis/
    │       ├── neutral_angle.png        ← 관절 각도 시각화
    │       └── max_selected_angle.png
    ├── landmark_json/
    │   ├── neutral.json
    │   ├── max_candidate_1.json
    │   ├── max_candidate_2.json
    │   └── max_selected.json
    └── rom_analysis_result.json         ← ★ 최종 ROM 결과

디렉토리 구조:
    mediapipe_rom_webcam_pipeline/
    ├── main.py                       ← 이 파일
    ├── core/
    │   ├── landmarks.py              ← BlazePose 33 SSOT
    │   ├── landmark_extractor.py     ← VIDEO + IMAGE 모드 랜드마크 추출
    │   ├── angle_engine.py           ← 3D 각도 계산 엔진
    │   └── visualizer.py             ← 각도 캔버스 + SVG
    ├── pipeline/
    │   ├── snapshot_rom_pipeline.py  ← ★ 스냅샷 ROM 파이프라인 (기본)
    │   └── rom_pipeline.py           ← 전체 프레임 처리 파이프라인
    └── videos/                       ← 입력 비디오 저장 폴더

모델 파일:
    프로젝트 루트/models/pose_landmarker_full.task (9MB)
    다운로드:
    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -OutFile "models/pose_landmarker_full.task"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── sys.path 설정 ─────────────────────────────────────────────────────────────
# 이 파일 위치: src/vision_ai/mediapipe_rom_webcam_pipeline/main.py
# 실행 방법 2가지 모두 지원:
#   (A) 프로젝트 루트에서: python src/vision_ai/mediapipe_rom_webcam_pipeline/main.py
#   (B) 이 폴더 안에서:   cd src/vision_ai/mediapipe_rom_webcam_pipeline && python main.py
_THIS_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = (_THIS_DIR / ".." / ".." / "..").resolve()

# (A) 프로젝트 루트를 sys.path에 추가
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# (B) 이 폴더 자체를 sys.path에 추가 (상대 import 지원)
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ── 파이프라인 임포트 ────────────────────────────────────────────────────────
try:
    from src.vision_ai.mediapipe_rom_webcam_pipeline.pipeline.rom_pipeline import run_video_pipeline
    from src.vision_ai.mediapipe_rom_webcam_pipeline.pipeline.snapshot_rom_pipeline import run_snapshot_rom_pipeline
    from src.vision_ai.mediapipe_rom_webcam_pipeline.core.angle_engine import VISIBILITY_THRESHOLD
except ImportError:
    from pipeline.rom_pipeline import run_video_pipeline              # type: ignore
    from pipeline.snapshot_rom_pipeline import run_snapshot_rom_pipeline  # type: ignore
    from core.angle_engine import VISIBILITY_THRESHOLD                # type: ignore

# ── 경로 상수 ────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH  = _PROJECT_ROOT / "models" / "pose_landmarker_full.task"
DEFAULT_VIDEOS_DIR  = _THIS_DIR / "videos"
DEFAULT_RESULTS_DIR = _THIS_DIR / "results"

# 지원하는 비디오 확장자
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


# ─────────────────────────────────────────────────────────────────────────────
# CLI 파서
# ─────────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Tro-Fit Vision AI — MediaPipe ROM 분석 파이프라인",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "예시:\n"
            "  # [snapshot 모드 - 기본] 4초 영상, 0.5s 중립, 끝-1s/끝-0.5s 최대 자동 캡처\n"
            "  python main.py --video videos/shoulder_test.mp4\n\n"
            "  # [snapshot 모드] 타임스탬프 직접 지정 (중립 0.5s, 최대 3.0s / 3.5s)\n"
            "  python main.py --video videos/shoulder_test.mp4 --neutral-ts 0.5 --max-ts 3.0,3.5\n\n"
            "  # [full 모드] 비디오 전체 처리 + 프리뷰\n"
            "  python main.py --video videos/squat.mp4 --mode full --preview\n\n"
            "  # [full 모드] 처음 100프레임만 테스트\n"
            "  python main.py --video videos/test.mp4 --mode full --max-frames 100\n"
        ),
    )

    # ── 입력 소스 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--video", "-v",
        type=str, default=None, metavar="PATH",
        help="처리할 비디오 파일 경로. 미지정 시 --videos-dir 폴더의 모든 비디오 처리.",
    )
    parser.add_argument(
        "--videos-dir",
        type=str, default=str(DEFAULT_VIDEOS_DIR), metavar="DIR",
        help=f"비디오 폴더 경로 (기본값: {DEFAULT_VIDEOS_DIR})",
    )

    # ── 모드 선택 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        type=str, default="snapshot", choices=["snapshot", "full"],
        help=(
            "파이프라인 모드 (기본값: snapshot)\n"
            "  snapshot: 특정 타임스탬프 3장 스냅샷으로 ROM 측정 (권장, 빠름)\n"
            "  full:     비디오 전체 프레임 순차 처리 (느림, 연속 ROM 추적)"
        ),
    )

    # ── snapshot 모드 전용 옵션 ────────────────────────────────────────
    parser.add_argument(
        "--neutral-ts",
        type=float, default=0.5, metavar="SEC",
        help="[snapshot] 중립 자세 캡처 타임스탬프 (초, 기본값: 0.5)",
    )
    parser.add_argument(
        "--max-ts",
        type=str, default="auto", metavar="SEC[,SEC]",
        help=(
            "[snapshot] 최대 자세 후보 타임스탬프 (초)\n"
            "  auto:    자동 계산 (영상 끝-1.0s, 끝-0.5s)\n"
            "  수동 예: 3.0,3.5 (쉼표로 구분, 최소 1개)"
        ),
    )

    # ── 모델 ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        type=str, default=str(DEFAULT_MODEL_PATH), metavar="PATH",
        help=f"pose_landmarker_full.task 파일 경로 (기본값: {DEFAULT_MODEL_PATH})",
    )

    # ── 출력 ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir", "-o",
        type=str, default=str(DEFAULT_RESULTS_DIR), metavar="DIR",
        help=f"결과 저장 루트 디렉토리 (기본값: {DEFAULT_RESULTS_DIR})",
    )

    # ── 공통 옵션 ─────────────────────────────────────────────────────
    parser.add_argument(
        "--no-world",
        action="store_true", default=False,
        help="world_landmarks 대신 정규화 좌표로 각도 계산 (정확도 낮음, 기본: world 사용)",
    )
    parser.add_argument(
        "--threshold",
        type=float, default=VISIBILITY_THRESHOLD, metavar="F",
        help=f"visibility 임계값 (0.0~1.0, 기본값: {VISIBILITY_THRESHOLD})",
    )
    parser.add_argument(
        "--no-angle-img",
        action="store_true", default=False,
        help="각도 시각화 이미지 저장 비활성화",
    )

    # ── full 모드 전용 옵션 ───────────────────────────────────────────
    parser.add_argument(
        "--preview",
        action="store_true", default=False,
        help="[full] 실시간 프리뷰 창 표시 (q키: 종료)",
    )
    parser.add_argument(
        "--max-frames",
        type=int, default=0, metavar="N",
        help="[full] 최대 처리 프레임 수 (0=전체)",
    )
    parser.add_argument(
        "--no-picto",
        action="store_true", default=False,
        help="[full] SVG 픽토그래픽 저장 비활성화",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# 비디오 수집
# ─────────────────────────────────────────────────────────────────────────────
def _collect_videos(video_arg: str | None, videos_dir: Path) -> list[Path]:
    """처리할 비디오 파일 목록을 반환합니다."""
    if video_arg:
        p = Path(video_arg)
        if not p.exists():
            print(f"[ERROR] 비디오 파일을 찾을 수 없습니다: {p}")
            sys.exit(1)
        return [p]

    if not videos_dir.exists():
        print(f"[INFO] videos/ 폴더가 없습니다. 생성 중: {videos_dir}")
        videos_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[HINT] videos/ 폴더에 비디오 파일을 넣고 다시 실행하세요.\n"
            f"       지원 형식: {', '.join(sorted(_VIDEO_EXTENSIONS))}"
        )
        sys.exit(0)

    videos = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
    )

    if not videos:
        print(
            f"[INFO] videos/ 폴더에 비디오 파일이 없습니다: {videos_dir}\n"
            f"       지원 형식: {', '.join(sorted(_VIDEO_EXTENSIONS))}"
        )
        sys.exit(0)

    return videos


# ─────────────────────────────────────────────────────────────────────────────
# --max-ts 파싱 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def _parse_max_ts(max_ts_arg: str) -> list[float] | None:
    """
    --max-ts 인수를 파싱합니다.
    'auto' → None (파이프라인이 자동 계산)
    '3.0,3.5' → [3.0, 3.5]
    """
    if max_ts_arg.strip().lower() == "auto":
        return None
    try:
        parts = [float(x.strip()) for x in max_ts_arg.split(",") if x.strip()]
        if not parts:
            raise ValueError("빈 타임스탬프")
        return parts
    except ValueError:
        print(f"[ERROR] --max-ts 형식 오류: '{max_ts_arg}'\n"
              "       올바른 형식: auto 또는 3.0,3.5 (쉼표 구분 초 단위)")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Windows cp949 콘솔 인코딩 방어
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = _build_parser()
    args   = parser.parse_args()

    # 비디오 목록 수집
    videos_dir = Path(args.videos_dir)
    video_list = _collect_videos(args.video, videos_dir)

    mode = args.mode
    print(f"\n  Tro-Fit Vision AI — ROM 분석 파이프라인")
    print(f"  모드      : {mode.upper()}")
    print(f"  모델 파일 : {args.model}")
    print(f"  처리 대상 : {len(video_list)} 개 비디오")
    print(f"  world 좌표: {'사용' if not args.no_world else '미사용'}")
    print(f"  threshold : {args.threshold}")
    if mode == "snapshot":
        print(f"  중립 캡처 : {args.neutral_ts}s")
        print(f"  최대 캡처 : {args.max_ts} (auto = 끝-1.0s, 끝-0.5s 자동)")

    all_results = []
    for i, video_path in enumerate(video_list, 1):
        print(f"\n[{i}/{len(video_list)}] {video_path.name}")
        output_dir = Path(args.output_dir) / video_path.stem

        try:
            if mode == "snapshot":
                # ── Snapshot 모드: 3개 타임스탬프 스냅샷 기반 ROM ─────
                max_ts_list = _parse_max_ts(args.max_ts)
                result = run_snapshot_rom_pipeline(
                    video_path=video_path,
                    model_path=args.model,
                    output_dir=output_dir,
                    neutral_ts=args.neutral_ts,
                    max_ts_list=max_ts_list,
                    use_world=not args.no_world,
                    threshold=args.threshold,
                    save_images=not args.no_angle_img,
                )
            else:
                # ── Full 모드: 전체 프레임 순차 처리 ─────────────────
                result = run_video_pipeline(
                    video_path=video_path,
                    model_path=args.model,
                    output_dir=output_dir,
                    use_world=not args.no_world,
                    preview=args.preview,
                    max_frames=args.max_frames,
                    threshold=args.threshold,
                    save_angle_images=not args.no_angle_img,
                    save_pictographic=not args.no_picto,
                )

            all_results.append(result)

        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            continue
        except KeyboardInterrupt:
            print("\n[INFO] 사용자가 중단했습니다.")
            break

    print(f"\n[완료] 총 {len(all_results)}/{len(video_list)} 개 비디오 처리 완료.")


if __name__ == "__main__":
    main()
