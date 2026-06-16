"""
visualizer.py — 관절 각도 시각화 모듈
─────────────────────────────────────────────────────────────────────────────

책임:
  - 검은 배경 골격 + 노란색 각도 레이블 이미지 생성 (build_angle_canvas)
  - 레이블 겹침 방지 알고리즘 (_draw_label with collision detection)
  - SVG 픽토그래픽 생성 (generate_pictographic_svg)
  - 포즈별 시각화 이미지 저장 (render_angle_images)

설계 포인트:
  - 이 모듈은 순수 렌더링 레이어 → 파일 I/O는 render_angle_images만 수행
  - BlazePoseLandmark SSOT 사용
  - LEFT/RIGHT 색상 분리 → 임상 가독성

의존성:
  from core.landmarks import BlazePoseLandmark, BODY_CONNECTIONS, LEFT_LANDMARKS, RIGHT_LANDMARKS
  from core.angle_engine import PoseAngleReport
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.angle_engine import PoseAngleReport
from core.landmarks import (
    BODY_CONNECTIONS,
    BlazePoseLandmark,
    LEFT_LANDMARKS,
    RIGHT_LANDMARKS,
)

# ──────────────────────────────────────────────────────────────────────────────
# 시각화 상수
# ──────────────────────────────────────────────────────────────────────────────
CANVAS_W = 640
CANVAS_H = 640
PADDING  = 60

# 색상 (BGR)
COLOR_LEFT_PT  = (  0, 140, 255)   # 주황색 — 좌측 관절
COLOR_RIGHT_PT = (255, 200,   0)   # 하늘색 — 우측 관절
COLOR_OTHER_PT = (255, 255, 255)   # 흰색   — 중립 관절
COLOR_LINE     = (255, 255, 255)   # 흰색 연결선
COLOR_LABEL    = (  0, 255, 255)   # 노란색 각도 레이블

FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.5
FONT_THICKNESS = 1
LABEL_PADDING  = 4

# SVG 디자인 토큰
SVG_BG_COLOR = "#0D1117"
SVG_POSE_PALETTE: list[str] = [
    "#00E5FF",  # 0: Cyan
    "#FF2D78",  # 1: Hot Pink
    "#FFD600",  # 2: Yellow
    "#00E676",  # 3: Green
    "#FF6D00",  # 4: Orange
    "#D500F9",  # 5: Purple
]
SVG_STROKE_WIDTH_RATIO = 0.038
SVG_HEAD_RADIUS_RATIO  = 0.048
SVG_VISIBILITY_DEFAULT = 0.25

# 각도 레이블 표시 대상 관절 → vertex BPL 매핑
_ANGLE_VERTEX_MAP: dict[str, BlazePoseLandmark] = {
    "left_knee":      BlazePoseLandmark.LEFT_KNEE,
    "right_knee":     BlazePoseLandmark.RIGHT_KNEE,
    "left_elbow":     BlazePoseLandmark.LEFT_ELBOW,
    "right_elbow":    BlazePoseLandmark.RIGHT_ELBOW,
    "left_shoulder":  BlazePoseLandmark.LEFT_SHOULDER,
    "right_shoulder": BlazePoseLandmark.RIGHT_SHOULDER,
    "left_hip":       BlazePoseLandmark.LEFT_HIP,
    "right_hip":      BlazePoseLandmark.RIGHT_HIP,
    "left_ankle":     BlazePoseLandmark.LEFT_ANKLE,
    "right_ankle":    BlazePoseLandmark.RIGHT_ANKLE,
}


# ──────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def _norm_to_pixel(
    nx: float, ny: float,
    w: int, h: int,
    pad: int = PADDING,
) -> tuple[int, int]:
    """정규화 좌표(0~1) → 패딩이 적용된 캔버스 픽셀 좌표."""
    inner_w = w - 2 * pad
    inner_h = h - 2 * pad
    return (int(nx * inner_w) + pad, int(ny * inner_h) + pad)


def _draw_label(
    canvas:      np.ndarray,
    text:        str,
    pt:          tuple[int, int],
    drawn_boxes: list[tuple[int, int, int, int]] | None = None,
) -> None:
    """
    노란 글씨 + 반투명 검은 배경 박스로 각도 레이블을 그립니다.

    - 캔버스 경계 클리핑 → 화면 밖 텍스트 방지
    - drawn_boxes 충돌 감지 → 최대 30회 반복으로 겹침 방지
    """
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    x, y = pt

    if drawn_boxes is not None:
        box_h = th + LABEL_PADDING * 2
        for _ in range(30):
            x1 = x - LABEL_PADDING
            y1 = y - th - LABEL_PADDING
            x2 = x + tw + LABEL_PADDING
            y2 = y + LABEL_PADDING

            collision = any(
                x1 < bx2 and x2 > bx1 and y1 < by2 and y2 > by1
                for bx1, by1, bx2, by2 in drawn_boxes
            )
            if not collision:
                break
            y += int(box_h * 0.8)

    # 경계 클리핑
    x = max(LABEL_PADDING, min(x, canvas.shape[1] - tw - LABEL_PADDING * 2))
    y = max(th + LABEL_PADDING, min(y, canvas.shape[0] - LABEL_PADDING))

    x1f = x - LABEL_PADDING
    y1f = y - th - LABEL_PADDING
    x2f = x + tw + LABEL_PADDING
    y2f = y + LABEL_PADDING

    if drawn_boxes is not None:
        drawn_boxes.append((x1f, y1f, x2f, y2f))

    cv2.rectangle(canvas, (x1f, y1f), (x2f, y2f), (30, 30, 30), cv2.FILLED)
    cv2.putText(canvas, text, (x, y), FONT, FONT_SCALE, COLOR_LABEL, FONT_THICKNESS, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# 각도 시각화 캔버스 생성
# ──────────────────────────────────────────────────────────────────────────────
def build_angle_canvas(
    raw_pose:     dict,
    angle_report: PoseAngleReport,
    canvas_w:     int = CANVAS_W,
    canvas_h:     int = CANVAS_H,
) -> np.ndarray:
    """
    단일 포즈의 정규화 좌표를 사용하여 검은 배경 골격 + 각도 레이블 이미지를 생성합니다.

    Args:
        raw_pose:     extract_frame_record()의 poses[i] 딕셔너리
        angle_report: analyze_pose()가 반환한 PoseAngleReport
        canvas_w:     출력 캔버스 너비
        canvas_h:     출력 캔버스 높이

    Returns:
        np.ndarray: BGR 시각화 이미지
    """
    canvas    = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    landmarks = raw_pose["landmarks"]

    # ── 1. 좌표 맵 구성 (visibility 0.2 이상만) ──────────────────────────
    coords: dict[int, tuple[int, int]] = {}
    for lm in BlazePoseLandmark:
        key = lm.json_key()
        if key not in landmarks:
            continue
        d = landmarks[key]
        if d.get("visibility", 1.0) < 0.2:
            continue
        coords[int(lm)] = _norm_to_pixel(d["x"], d["y"], canvas_w, canvas_h)

    # ── 2. BODY_CONNECTIONS 기반 흰색 연결선 ─────────────────────────────
    for start_lm, end_lm in BODY_CONNECTIONS:
        s, e = int(start_lm), int(end_lm)
        if s in coords and e in coords:
            cv2.line(canvas, coords[s], coords[e], COLOR_LINE, 2, lineType=cv2.LINE_AA)

    # ── 3. 관절 포인트 그리기 ─────────────────────────────────────────────
    for lm in BlazePoseLandmark:
        idx = int(lm)
        if idx not in coords:
            continue
        pt = coords[idx]
        inner_color = (
            COLOR_LEFT_PT  if lm in LEFT_LANDMARKS  else
            COLOR_RIGHT_PT if lm in RIGHT_LANDMARKS else
            COLOR_OTHER_PT
        )
        cv2.circle(canvas, pt, 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)  # 흰 테두리
        cv2.circle(canvas, pt, 4, inner_color,     -1, lineType=cv2.LINE_AA)  # 컬러 내부

    # ── 4. 각도 레이블 표시 ───────────────────────────────────────────────
    drawn_boxes: list[tuple[int, int, int, int]] = []
    for joint_result in angle_report.joints:
        if not joint_result.reliable or joint_result.angle_deg is None:
            continue
        vertex_lm = _ANGLE_VERTEX_MAP.get(joint_result.joint)
        if vertex_lm is None or int(vertex_lm) not in coords:
            continue

        short_name = joint_result.joint.replace("_", " ")
        label_text = f"{short_name}: {joint_result.angle_deg:.1f}deg"

        lx, ly = coords[int(vertex_lm)]
        if "left" in joint_result.joint:
            (tw, _), _ = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)
            offset_x = lx - tw - 12
        else:
            offset_x = lx + 8

        _draw_label(canvas, label_text, (offset_x, ly - 10), drawn_boxes)

    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# 각도 시각화 이미지 파일 저장
# ──────────────────────────────────────────────────────────────────────────────
def render_angle_images(
    name:     str,
    raw_data: dict,
    reports:  list[PoseAngleReport],
    out_dir:  Path,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
) -> list[Path]:
    """
    한 프레임/비디오 세그먼트의 모든 포즈에 대해 각도 시각화 이미지를 저장합니다.

    Args:
        name:     파일명 접두어 (예: "frame_0042", "segment_squat")
        raw_data: extract_frame_record() 반환값
        reports:  analyze_pose() 반환값 목록
        out_dir:  저장 디렉토리

    Returns:
        list[Path]: 저장된 파일 경로 목록
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for i, (raw_pose, report) in enumerate(zip(raw_data.get("poses", []), reports)):
        img = build_angle_canvas(raw_pose, report, canvas_w, canvas_h)
        suffix = "" if len(reports) == 1 else f"_pose{i}"
        filename = f"{name}{suffix}_angle.png"
        out_path = out_dir / filename
        cv2.imwrite(str(out_path), img)
        saved.append(out_path)

    return saved


# ──────────────────────────────────────────────────────────────────────────────
# SVG 픽토그래픽 생성
# ──────────────────────────────────────────────────────────────────────────────
def _pose_color(pose_index: int) -> str:
    return SVG_POSE_PALETTE[pose_index % len(SVG_POSE_PALETTE)]


def _build_svg_coords(
    landmarks:           dict,
    scale_x:             float,
    scale_y:             float,
    visibility_threshold: float,
) -> dict[BlazePoseLandmark, tuple[float, float]]:
    """landmarks dict에서 visibility >= threshold인 관절의 픽셀 좌표를 반환."""
    coords: dict[BlazePoseLandmark, tuple[float, float]] = {}
    for bpl in BlazePoseLandmark:
        data = landmarks.get(bpl.json_key())
        if data is None:
            continue
        if data.get("visibility", 0) < visibility_threshold:
            continue
        coords[bpl] = (
            data["pixel_x"] * scale_x,
            data["pixel_y"] * scale_y,
        )
    return coords


def _line_path(a: tuple[float, float], b: tuple[float, float]) -> str:
    return f"M {a[0]:.2f},{a[1]:.2f} L {b[0]:.2f},{b[1]:.2f}"


def generate_pictographic_svg(
    poses_data:           list[dict],
    image_width:          int,
    image_height:         int,
    output_path:          str | Path,
    svg_width:            Optional[int] = None,
    svg_height:           Optional[int] = None,
    visibility_threshold: float = SVG_VISIBILITY_DEFAULT,
) -> str:
    """
    extract_frame_record()의 poses 리스트로부터 SVG 벡터 픽토그래픽을 생성하고 저장합니다.

    특징:
      - 짙은 다크 네이비 배경 (#0D1117)
      - 포즈 인덱스별 색상 팔레트 (Cyan → Hot Pink → Yellow → ...)
      - stroke-linecap="round" → 굵고 동글동글한 선
      - BODY_CONNECTIONS만 사용 → 얼굴 세부선·손가락 제외, 깔끔한 실루엣
      - 코(NOSE) 위치에 머리 원 → 포즈 색으로 채워 통일감

    Returns:
        str: 저장된 SVG 문자열
    """
    out_w = svg_width  or image_width
    out_h = svg_height or image_height
    scale_x = out_w / image_width  if image_width  else 1.0
    scale_y = out_h / image_height if image_height else 1.0

    ref    = min(out_w, out_h)
    sw     = max(4.0, ref * SVG_STROKE_WIDTH_RATIO)
    head_r = max(8.0, ref * SVG_HEAD_RADIUS_RATIO)

    svg = ET.Element("svg", {
        "xmlns":   "http://www.w3.org/2000/svg",
        "width":   str(out_w),
        "height":  str(out_h),
        "viewBox": f"0 0 {out_w} {out_h}",
    })
    ET.SubElement(svg, "rect", {
        "width": str(out_w), "height": str(out_h), "fill": SVG_BG_COLOR,
    })

    g_bones = ET.SubElement(svg, "g", {
        "stroke-linecap": "round", "stroke-linejoin": "round", "fill": "none",
    })
    g_heads = ET.SubElement(svg, "g")

    for pose in poses_data:
        pose_idx  = pose.get("pose_index", 0)
        color     = _pose_color(pose_idx)
        landmarks = pose.get("landmarks", {})
        coords    = _build_svg_coords(landmarks, scale_x, scale_y, visibility_threshold)

        for si, ei in BODY_CONNECTIONS:
            if si not in coords or ei not in coords:
                continue
            ET.SubElement(g_bones, "path", {
                "d": _line_path(coords[si], coords[ei]),
                "stroke": color,
                "stroke-width": f"{sw:.2f}",
            })

        nose_key = BlazePoseLandmark.NOSE
        if nose_key in coords:
            nx, ny = coords[nose_key]
            ET.SubElement(g_heads, "circle", {
                "cx": f"{nx:.2f}", "cy": f"{ny:.2f}",
                "r": f"{head_r:.2f}", "fill": color,
            })

    ET.indent(svg, space="  ")
    svg_bytes = ET.tostring(svg, encoding="unicode", xml_declaration=False)
    svg_string = f'<?xml version="1.0" encoding="UTF-8"?>\n{svg_bytes}\n'

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg_string, encoding="utf-8")
    return svg_string
