"""
pictographic_generator.py
─────────────────────────────────────────────────────────────────────────────
관절 좌표(JSON 형식)로부터 벡터(SVG) 픽토그래픽 이미지를 생성하는 모듈.

특징:
  - 출력 포맷    : SVG (벡터 이미지, 무한 확대 가능)
  - 관절 마커 없음: 선(stroke)만으로 인체를 표현 (얼굴·손가락 세부선 제외)
  - 굵고 동글동글한 선: stroke-linecap="round" + stroke-linejoin="round"
                        stroke-width 를 해상도 대비 3~4% 수준으로 설정
  - 포즈별 단일 색상 : pose_index 순서대로 Cyan → Pink → Yellow → Green …
  - 머리          : 코(idx=0) 기준 원, 포즈 색으로 채워 통일감 유지
  - 배경          : 짙은 다크 네이비 (#0D1117)
  - 눈·귀·입·손가락 같은 세밀한 연결선은 제외 → 깔끔한 실루엣

사용법:
  from pictographic_generator import generate_pictographic_svg

  generate_pictographic_svg(
      poses_data=data["poses"],
      image_width=data["image_width"],
      image_height=data["image_height"],
      output_path="output.svg",
  )
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# BlazePose 33개 랜드마크 이름 (인덱스 순서)
# ──────────────────────────────────────────────────────────────────────────────
LANDMARK_NAMES: list[str] = [
    "nose",              # 0
    "left_eye_inner",    # 1
    "left_eye",          # 2
    "left_eye_outer",    # 3
    "right_eye_inner",   # 4
    "right_eye",         # 5
    "right_eye_outer",   # 6
    "left_ear",          # 7
    "right_ear",         # 8
    "mouth_left",        # 9
    "mouth_right",       # 10
    "left_shoulder",     # 11
    "right_shoulder",    # 12
    "left_elbow",        # 13
    "right_elbow",       # 14
    "left_wrist",        # 15
    "right_wrist",       # 16
    "left_pinky",        # 17
    "right_pinky",       # 18
    "left_index",        # 19
    "right_index",       # 20
    "left_thumb",        # 21
    "right_thumb",       # 22
    "left_hip",          # 23
    "right_hip",         # 24
    "left_knee",         # 25
    "right_knee",        # 26
    "left_ankle",        # 27
    "right_ankle",       # 28
    "left_heel",         # 29
    "right_heel",        # 30
    "left_foot_index",   # 31
    "right_foot_index",  # 32
]

# ──────────────────────────────────────────────────────────────────────────────
# 픽토그래픽용 뼈대 연결 (얼굴 세부 / 손가락 제외)
#   눈·귀·입·손가락 연결을 제거해 '민트색 점' 현상 원천 차단
#   발은 발목→발끝 단일 선만 유지
# ──────────────────────────────────────────────────────────────────────────────
BODY_CONNECTIONS: list[tuple[int, int]] = [
    # 어깨 (torso top)
    (11, 12),
    # 왼팔
    (11, 13), (13, 15),
    # 오른팔
    (12, 14), (14, 16),
    # 몸통
    (11, 23), (12, 24), (23, 24),
    # 왼다리
    (23, 25), (25, 27),
    # 오른다리
    (24, 26), (26, 28),
    # 왼발 (발목 → 발끝)
    (27, 31),
    # 오른발
    (28, 32),
]

# ──────────────────────────────────────────────────────────────────────────────
# 디자인 토큰
# ──────────────────────────────────────────────────────────────────────────────
BG_COLOR = "#0D1117"  # 짙은 다크 네이비 배경

# 포즈 인덱스 순서대로 할당되는 색상 팔레트
POSE_PALETTE: list[str] = [
    "#00E5FF",  # 0: Cyan
    "#FF2D78",  # 1: Hot Pink
    "#FFD600",  # 2: Yellow
    "#00E676",  # 3: Green
    "#FF6D00",  # 4: Orange
    "#D500F9",  # 5: Purple
]

# 비율 상수 (min(width, height) 기준)
STROKE_WIDTH_RATIO = 0.038   # 굵은 선: 이미지 짧은 변의 3.8%
HEAD_RADIUS_RATIO  = 0.048   # 머리 반지름: 이미지 짧은 변의 4.8%

# 기본 visibility 임계값
DEFAULT_VISIBILITY = 0.25


# ──────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def _pose_color(pose_index: int) -> str:
    """포즈 인덱스로 색상을 반환 (팔레트 순환)."""
    return POSE_PALETTE[pose_index % len(POSE_PALETTE)]


def _build_coords(
    landmarks: dict,
    scale_x: float,
    scale_y: float,
    visibility_threshold: float,
) -> dict[int, tuple[float, float]]:
    """
    landmarks dict 에서 visibility >= threshold 인 관절의 픽셀 좌표를 반환.
    """
    coords: dict[int, tuple[float, float]] = {}
    for idx, name in enumerate(LANDMARK_NAMES):
        lm = landmarks.get(name)
        if lm is None:
            continue
        if lm.get("visibility", 0) < visibility_threshold:
            continue
        coords[idx] = (
            lm["pixel_x"] * scale_x,
            lm["pixel_y"] * scale_y,
        )
    return coords


def _line_path(a: tuple[float, float], b: tuple[float, float]) -> str:
    """두 점 사이 직선 SVG path (round linecap으로 끝이 자동으로 둥글어짐)."""
    return f"M {a[0]:.2f},{a[1]:.2f} L {b[0]:.2f},{b[1]:.2f}"


# ──────────────────────────────────────────────────────────────────────────────
# 핵심 공개 함수
# ──────────────────────────────────────────────────────────────────────────────
def landmarks_to_svg(
    poses_data: list[dict],
    image_width: int,
    image_height: int,
    svg_width:  Optional[int] = None,
    svg_height: Optional[int] = None,
    visibility_threshold: float = DEFAULT_VISIBILITY,
) -> str:
    """
    extract_landmarks_json() 의 반환값(poses_data)으로부터
    SVG 벡터 픽토그래픽 이미지 문자열을 생성합니다.

    Args:
        poses_data           : extract_landmarks_json() 반환값
        image_width          : 원본 이미지 너비 (pixel)
        image_height         : 원본 이미지 높이 (pixel)
        svg_width            : SVG 출력 너비  (None → image_width)
        svg_height           : SVG 출력 높이 (None → image_height)
        visibility_threshold : 이 값 미만의 관절은 렌더링하지 않음

    Returns:
        str: 완성된 SVG 문자열
    """
    out_w = svg_width  if svg_width  else image_width
    out_h = svg_height if svg_height else image_height

    scale_x = out_w / image_width  if image_width  else 1.0
    scale_y = out_h / image_height if image_height else 1.0

    ref   = min(out_w, out_h)
    sw    = max(4.0, ref * STROKE_WIDTH_RATIO)   # 선 굵기
    head_r = max(8.0, ref * HEAD_RADIUS_RATIO)   # 머리 반지름

    # ── SVG 루트 ──────────────────────────────────────────────────────
    svg = ET.Element("svg", {
        "xmlns":   "http://www.w3.org/2000/svg",
        "width":   str(out_w),
        "height":  str(out_h),
        "viewBox": f"0 0 {out_w} {out_h}",
    })

    # 배경 사각형
    ET.SubElement(svg, "rect", {
        "width":  str(out_w),
        "height": str(out_h),
        "fill":   BG_COLOR,
    })

    # ── 뼈대 레이어 (fill=none, round cap/join) ────────────────────────
    g_bones = ET.SubElement(svg, "g", {
        "stroke-linecap":  "round",
        "stroke-linejoin": "round",
        "fill":            "none",
    })

    # ── 머리 레이어 (뼈대 위에 렌더링) ────────────────────────────────
    g_heads = ET.SubElement(svg, "g")

    for pose in poses_data:
        pose_idx  = pose.get("pose_index", 0)
        color     = _pose_color(pose_idx)
        landmarks = pose.get("landmarks", {})
        coords    = _build_coords(landmarks, scale_x, scale_y, visibility_threshold)

        # 뼈대 연결선 (얼굴/손가락 제외된 BODY_CONNECTIONS만 사용)
        for (si, ei) in BODY_CONNECTIONS:
            if si not in coords or ei not in coords:
                continue
            ET.SubElement(g_bones, "path", {
                "d":            _line_path(coords[si], coords[ei]),
                "stroke":       color,
                "stroke-width": f"{sw:.2f}",
            })

        # 머리: 코(idx=0) 위치에 채워진 원 (관절 마커가 아닌 '머리' 표현)
        if 0 in coords:
            nx, ny = coords[0]
            ET.SubElement(g_heads, "circle", {
                "cx":   f"{nx:.2f}",
                "cy":   f"{ny:.2f}",
                "r":    f"{head_r:.2f}",
                "fill": color,          # 포즈 색으로 채움 → 통일감
            })

    # ── SVG 직렬화 ─────────────────────────────────────────────────────
    ET.indent(svg, space="  ")
    svg_bytes = ET.tostring(svg, encoding="unicode", xml_declaration=False)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{svg_bytes}\n'


def save_svg(svg_string: str, output_path: str) -> None:
    """SVG 문자열을 파일로 저장합니다."""
    with open(output_path, mode="w", encoding="utf-8") as f:
        f.write(svg_string)


def generate_pictographic_svg(
    poses_data: list[dict],
    image_width: int,
    image_height: int,
    output_path: str,
    svg_width:  Optional[int] = None,
    svg_height: Optional[int] = None,
    visibility_threshold: float = DEFAULT_VISIBILITY,
) -> str:
    """
    SVG 픽토그래픽 이미지를 생성하고 파일로 저장하는 단축 함수.

    Returns:
        str: 저장된 SVG 문자열
    """
    svg_string = landmarks_to_svg(
        poses_data=poses_data,
        image_width=image_width,
        image_height=image_height,
        svg_width=svg_width,
        svg_height=svg_height,
        visibility_threshold=visibility_threshold,
    )
    save_svg(svg_string, output_path)
    return svg_string
