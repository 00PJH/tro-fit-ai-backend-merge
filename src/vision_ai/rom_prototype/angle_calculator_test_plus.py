"""
angle_calculator_test_plus.py — 관절 각도 계산 및 시각화 모듈 (확장 버전)
==========================================================

[추가된 관절]
  - 고관절 굴곡 (Hip Flexion)
  - 발목 배측굴곡 (Ankle Dorsiflexion)
  - 어깨 외전 (Shoulder Abduction)
  - 팔꿈치 굴곡 (Elbow Flexion)

[저장 구조]
  results/
  ├── joint_33/
  │   ├── landmark_json/   : pose_test.py 가 생성 (개별/통합 landmark JSON + CSV)
  │   └── joint_img/       : pose_test.py 가 생성 (관절 추출 결과 이미지)
  ├── pictographic/        : pose_test.py 가 생성 (SVG 픽토그래픽)
  └── angle/
      ├── angle_json/
      │   ├── test_1_angle.json      : 이미지별 각도 결과
      │   ├── test_2_angle.json
      │   ├── test_3_angle.json
      │   └── angle_all.json         : 통합 결과
      └── angle_img/
          ├── test_1_angle_vis.png   : 각도 시각화 이미지 (검은 배경 + 골격 + 노란 각도 레이블)
          ├── test_2_angle_vis.png
          └── test_3_angle_vis.png

[각도 정의]
  - 무릎(Knee)               : Hip → Knee → Ankle
  - 팔꿈치(Elbow Flexion)    : Shoulder → Elbow → Wrist
  - 어깨(Shoulder Abduction) : Elbow → Shoulder → Hip
  - 고관절(Hip Flexion)      : Shoulder → Hip → Knee
  - 발목(Ankle Dorsiflexion) : Knee → Ankle → Foot Index

[시각화 이미지 스펙]
  - 검은 배경 (Black canvas)
  - 좌측 랜드마크 : 주황색 점 (BGR 0,140,255)
  - 우측 랜드마크 : 하늘색 점 (BGR 255,200,0)
  - 연결선        : 흰색 (255,255,255)
  - 각도 레이블   : 노란색 (0,255,255 BGR) — 각 관절에 표시
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# ── 절대 경로 기반 프로젝트 루트 추가 ─────────────────────────────────────
_THIS_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = (_THIS_DIR / ".." / ".." / "..").resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.vision_ai.media_pipe_test.landmarks import (
    BlazePoseLandmark as BPL,
    BODY_CONNECTIONS,
    LEFT_LANDMARKS,
    RIGHT_LANDMARKS,
)

# ─────────────────────────────────────────────────────────────────────────────
# 상수 및 경로 설정
# ─────────────────────────────────────────────────────────────────────────────

VISIBILITY_THRESHOLD: float = 0.65     # 이 미만 visibility → 신뢰 불가

# 입력 데이터: pose_test.py 가 생성한 landmark JSON 위치
#   results/joint_33/landmark_json/test_X_landmarks.json
RESULTS_DIR   = _THIS_DIR.parent / "img_test" / "results"
JOINT33_DIR   = RESULTS_DIR / "joint_33"
LM_JSON_DIR   = JOINT33_DIR / "landmark_json"   # landmark JSON 전용 폴더
ANGLE_DIR      = RESULTS_DIR / "angle"
ANGLE_JSON_DIR = ANGLE_DIR / "angle_json"
ANGLE_IMG_DIR  = ANGLE_DIR / "angle_img"

TEST_FILES: dict[str, Path] = {
    "test_1": LM_JSON_DIR / "test_1_landmarks.json",
    "test_2": LM_JSON_DIR / "test_2_landmarks.json",
    "test_3": LM_JSON_DIR / "test_3_landmarks.json",
}

# 시각화 이미지 캔버스 크기 (정규화 좌표를 이 크기로 스케일)
CANVAS_W = 640
CANVAS_H = 640
PADDING  = 60   # 외곽 여백 (정규화 좌표가 0~1 이므로 여백으로 공간 확보)

# 색상 (BGR)
COLOR_LEFT_PT   = (0,   140, 255)   # 주황색 — 좌측 관절
COLOR_RIGHT_PT  = (255, 200,   0)   # 하늘색 — 우측 관절
COLOR_OTHER_PT  = (255, 255, 255)   # 흰색   — 코 등 중립 관절
COLOR_LINE      = (255, 255, 255)   # 흰색 연결선
COLOR_LABEL     = (  0, 255, 255)   # 노란색 각도 레이블
COLOR_LABEL_BG  = (  0,   0,   0)   # 검은 배경 (캔버스 자체)

FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.5
FONT_THICKNESS  = 1
LABEL_PADDING   = 4   # 레이블 텍스트 배경 패딩

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────────────────────────────────────

class LandmarkPoint(NamedTuple):
    """단일 랜드마크의 3D 좌표와 신뢰도."""
    x: float
    y: float
    z: float
    visibility: float
    name: str


@dataclass
class JointAngleResult:
    """관절 하나의 각도 계산 결과."""
    joint:        str
    angle_deg:    float | None   # None = visibility 미달
    reliable:     bool
    point_a:      str   = ""
    vertex:       str   = ""
    point_c:      str   = ""
    visibility_a: float = 0.0
    visibility_v: float = 0.0
    visibility_c: float = 0.0

    def display(self) -> str:
        if not self.reliable:
            return (f"  [{self.joint:25s}] UNRELIABLE "
                    f"(vis: {self.visibility_a:.2f} / {self.visibility_v:.2f} / {self.visibility_c:.2f})")
        return (f"  [{self.joint:25s}] {self.angle_deg:7.2f} deg  "
                f"({self.point_a} -- {self.vertex} -- {self.point_c})")


@dataclass
class PoseAngleReport:
    """한 Pose의 전체 관절 각도 리포트."""
    pose_index: int
    joints: list[JointAngleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pose_index": self.pose_index,
            "joints": [asdict(j) for j in self.joints],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 핵심 수학 엔진
# ─────────────────────────────────────────────────────────────────────────────

def calculate_angle_3d(
    point_a: LandmarkPoint,
    vertex:  LandmarkPoint,
    point_c: LandmarkPoint,
) -> float:
    """
    3D 벡터 내적으로 vertex를 꼭짓점으로 하는 A-Vertex-C 사이각(도)을 반환합니다.
    수치 안정성: 동일 좌표 방어(1e-9), arccos 클리핑[-1,1].
    """
    a = np.array([point_a.x, point_a.y, point_a.z], dtype=np.float64)
    v = np.array([vertex.x,  vertex.y,  vertex.z],  dtype=np.float64)
    c = np.array([point_c.x, point_c.y, point_c.z], dtype=np.float64)

    va, vc = a - v, c - v
    n_va, n_vc = np.linalg.norm(va), np.linalg.norm(vc)

    if n_va < 1e-9 or n_vc < 1e-9:
        return 0.0

    cos_a = float(np.clip(np.dot(va, vc) / (n_va * n_vc), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


# ─────────────────────────────────────────────────────────────────────────────
# 랜드마크 파싱
# ─────────────────────────────────────────────────────────────────────────────

def _parse(raw: dict, name: str) -> LandmarkPoint:
    return LandmarkPoint(
        x=float(raw["x"]), y=float(raw["y"]), z=float(raw["z"]),
        visibility=float(raw.get("visibility", 1.0)), name=name,
    )

def get_lm(landmarks: dict[str, dict], lm: BPL) -> LandmarkPoint:
    """BlazePoseLandmark enum으로 랜드마크를 가져옵니다 (SSOT 연동)."""
    key = lm.json_key()
    return _parse(landmarks[key], key)


# ─────────────────────────────────────────────────────────────────────────────
# 관절별 각도 계산
# ─────────────────────────────────────────────────────────────────────────────

def _compute(
    joint_name: str,
    lm_a: LandmarkPoint, lm_v: LandmarkPoint, lm_c: LandmarkPoint,
    threshold: float = VISIBILITY_THRESHOLD,
) -> JointAngleResult:
    """visibility 검증 후 각도 계산. 미달 시 reliable=False 반환."""
    res = JointAngleResult(
        joint=joint_name, angle_deg=None, reliable=False,
        point_a=lm_a.name, vertex=lm_v.name, point_c=lm_c.name,
        visibility_a=lm_a.visibility, visibility_v=lm_v.visibility, visibility_c=lm_c.visibility,
    )
    if min(lm_a.visibility, lm_v.visibility, lm_c.visibility) < threshold:
        return res
    res.reliable  = True
    res.angle_deg = calculate_angle_3d(lm_a, lm_v, lm_c)
    return res


def compute_knee_angles(landmarks: dict, threshold=VISIBILITY_THRESHOLD) -> list[JointAngleResult]:
    """무릎: Hip — Knee — Ankle (좌우)"""
    return [
        _compute(f"{side}_knee",
                 get_lm(landmarks, hip), get_lm(landmarks, knee), get_lm(landmarks, ankle),
                 threshold)
        for side, hip, knee, ankle in [
            ("left",  BPL.LEFT_HIP,  BPL.LEFT_KNEE,  BPL.LEFT_ANKLE),
            ("right", BPL.RIGHT_HIP, BPL.RIGHT_KNEE, BPL.RIGHT_ANKLE),
        ]
    ]


def compute_elbow_angles(landmarks: dict, threshold=VISIBILITY_THRESHOLD) -> list[JointAngleResult]:
    """팔꿈치 굴곡: Shoulder — Elbow — Wrist (좌우)"""
    return [
        _compute(f"{side}_elbow",
                 get_lm(landmarks, shoulder), get_lm(landmarks, elbow), get_lm(landmarks, wrist),
                 threshold)
        for side, shoulder, elbow, wrist in [
            ("left",  BPL.LEFT_SHOULDER,  BPL.LEFT_ELBOW,  BPL.LEFT_WRIST),
            ("right", BPL.RIGHT_SHOULDER, BPL.RIGHT_ELBOW, BPL.RIGHT_WRIST),
        ]
    ]


def compute_shoulder_angles(landmarks: dict, threshold=VISIBILITY_THRESHOLD) -> list[JointAngleResult]:
    """어깨 외전/거상: Elbow — Shoulder — Hip (좌우)"""
    return [
        _compute(f"{side}_shoulder",
                 get_lm(landmarks, elbow), get_lm(landmarks, shoulder), get_lm(landmarks, hip),
                 threshold)
        for side, elbow, shoulder, hip in [
            ("left",  BPL.LEFT_ELBOW,  BPL.LEFT_SHOULDER,  BPL.LEFT_HIP),
            ("right", BPL.RIGHT_ELBOW, BPL.RIGHT_SHOULDER, BPL.RIGHT_HIP),
        ]
    ]


def compute_hip_angles(landmarks: dict, threshold=VISIBILITY_THRESHOLD) -> list[JointAngleResult]:
    """고관절 굴곡: Shoulder — Hip — Knee (좌우)"""
    return [
        _compute(f"{side}_hip",
                 get_lm(landmarks, shoulder), get_lm(landmarks, hip), get_lm(landmarks, knee),
                 threshold)
        for side, shoulder, hip, knee in [
            ("left",  BPL.LEFT_SHOULDER,  BPL.LEFT_HIP,  BPL.LEFT_KNEE),
            ("right", BPL.RIGHT_SHOULDER, BPL.RIGHT_HIP, BPL.RIGHT_KNEE),
        ]
    ]


def compute_ankle_angles(landmarks: dict, threshold=VISIBILITY_THRESHOLD) -> list[JointAngleResult]:
    """발목 배측굴곡: Knee — Ankle — Foot Index (좌우)"""
    return [
        _compute(f"{side}_ankle",
                 get_lm(landmarks, knee), get_lm(landmarks, ankle), get_lm(landmarks, foot_index),
                 threshold)
        for side, knee, ankle, foot_index in [
            ("left",  BPL.LEFT_KNEE,  BPL.LEFT_ANKLE,  BPL.LEFT_FOOT_INDEX),
            ("right", BPL.RIGHT_KNEE, BPL.RIGHT_ANKLE, BPL.RIGHT_FOOT_INDEX),
        ]
    ]


def analyze_pose(pose: dict) -> PoseAngleReport:
    """단일 포즈 딕셔너리 -> PoseAngleReport."""
    lm = pose["landmarks"]
    report = PoseAngleReport(pose_index=pose["pose_index"])
    report.joints.extend(compute_knee_angles(lm))
    report.joints.extend(compute_elbow_angles(lm))
    report.joints.extend(compute_shoulder_angles(lm))
    report.joints.extend(compute_hip_angles(lm))
    report.joints.extend(compute_ankle_angles(lm))
    return report


def analyze_file(json_path: Path) -> dict:
    """JSON 파일 하나를 분석하여 구조화된 결과 딕셔너리 반환."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    reports = [analyze_pose(p) for p in data.get("poses", [])]
    return {
        "source_file":          json_path.name,
        "image_width":          data.get("image_width"),
        "image_height":         data.get("image_height"),
        "num_poses_detected":   data.get("num_poses_detected", len(reports)),
        "visibility_threshold": VISIBILITY_THRESHOLD,
        "poses":                [r.to_dict() for r in reports],
        "_reports":             reports,   # 콘솔/시각화용 (저장 시 제외)
        "_raw":                 data,      # 시각화 좌표 참조용 (저장 시 제외)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 시각화: 검은 배경 골격 이미지 + 노란색 각도 레이블
# ─────────────────────────────────────────────────────────────────────────────

def _norm_to_pixel(nx: float, ny: float, w: int, h: int,
                   pad: int = PADDING) -> tuple[int, int]:
    """정규화 좌표(0~1) → 패딩이 적용된 캔버스 픽셀 좌표."""
    inner_w = w - 2 * pad
    inner_h = h - 2 * pad
    px = int(nx * inner_w) + pad
    py = int(ny * inner_h) + pad
    return (px, py)


def _draw_label(canvas: np.ndarray, text: str, pt: tuple[int, int], drawn_boxes: list[tuple[int, int, int, int]] | None = None) -> None:
    """
    노란 글씨 + 반투명 검은 배경 박스로 각도 레이블을 그립니다.
    텍스트가 캔버스 밖으로 나가지 않도록 자동 클리핑하며,
    drawn_boxes 리스트가 주어지면 기존 상자와 겹치지 않게 위치를 조정합니다.
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
            
            collision = False
            for (bx1, by1, bx2, by2) in drawn_boxes:
                if x1 < bx2 and x2 > bx1 and y1 < by2 and y2 > by1:
                    collision = True
                    break
            
            if not collision:
                break
                
            # 겹치면 아래로 이동
            y += int(box_h * 0.8)

    # 텍스트 박스가 캔버스를 벗어나지 않도록 최종 보정
    x = max(LABEL_PADDING, min(x, canvas.shape[1] - tw - LABEL_PADDING * 2))
    y = max(th + LABEL_PADDING, min(y, canvas.shape[0] - LABEL_PADDING))

    x1_final = x - LABEL_PADDING
    y1_final = y - th - LABEL_PADDING
    x2_final = x + tw + LABEL_PADDING
    y2_final = y + LABEL_PADDING

    if drawn_boxes is not None:
        drawn_boxes.append((x1_final, y1_final, x2_final, y2_final))

    # 검은 배경 사각형
    cv2.rectangle(
        canvas,
        (x1_final, y1_final),
        (x2_final, y2_final),
        (30, 30, 30), cv2.FILLED
    )
    # 노란 텍스트
    cv2.putText(canvas, text, (x, y), FONT, FONT_SCALE, COLOR_LABEL, FONT_THICKNESS, cv2.LINE_AA)


def build_angle_image(
    raw_pose: dict,
    angle_report: PoseAngleReport,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
) -> np.ndarray:
    """
    단일 포즈의 정규화 좌표를 사용하여 검은 배경 골격 + 각도 레이블 이미지를 생성합니다.
    """
    canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    landmarks = raw_pose["landmarks"]

    # ── 1. 좌표 맵 구성 (visibility 0.2 이상만) ──────────────────────────
    coords: dict[int, tuple[int, int]] = {}
    for lm in BPL:
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
    for lm in BPL:
        idx = int(lm)
        if idx not in coords:
            continue
        pt = coords[idx]
        if lm in LEFT_LANDMARKS:
            inner_color = COLOR_LEFT_PT
        elif lm in RIGHT_LANDMARKS:
            inner_color = COLOR_RIGHT_PT
        else:
            inner_color = COLOR_OTHER_PT

        cv2.circle(canvas, pt, 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)  # 흰 테두리
        cv2.circle(canvas, pt, 4, inner_color,     -1, lineType=cv2.LINE_AA)  # 컬러 내부

    # ── 4. 각도 레이블 표시 ───────────────────────────────────────────────
    # joint 이름 → vertex BPL 매핑
    vertex_map: dict[str, BPL] = {
        "left_knee":       BPL.LEFT_KNEE,
        "right_knee":      BPL.RIGHT_KNEE,
        "left_elbow":      BPL.LEFT_ELBOW,
        "right_elbow":     BPL.RIGHT_ELBOW,
        "left_shoulder":   BPL.LEFT_SHOULDER,
        "right_shoulder":  BPL.RIGHT_SHOULDER,
        "left_hip":        BPL.LEFT_HIP,
        "right_hip":       BPL.RIGHT_HIP,
        "left_ankle":      BPL.LEFT_ANKLE,
        "right_ankle":     BPL.RIGHT_ANKLE,
    }

    drawn_boxes: list[tuple[int, int, int, int]] = []

    for joint_result in angle_report.joints:
        if not joint_result.reliable or joint_result.angle_deg is None:
            continue
        vertex_lm = vertex_map.get(joint_result.joint)
        if vertex_lm is None:
            continue
        idx = int(vertex_lm)
        if idx not in coords:
            continue

        # 레이블 텍스트: "knee: 110.7deg" (기호는 아스키 안전 처리)
        short_name = joint_result.joint.replace("_", " ")   # e.g. "left knee"
        label_text = f"{short_name}: {joint_result.angle_deg:.1f}deg"

        # 관절 위치 기준 좌/우 분리 오프셋 설정
        lx, ly = coords[idx]
        
        if "left" in joint_result.joint:
            # 텍스트 크기를 미리 계산하여 관절의 왼쪽에 배치되도록 이동
            (tw, th), _ = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)
            offset_x = lx - tw - 12
        else:
            # 우측 관절은 관절의 오른쪽에 배치
            offset_x = lx + 8
            
        _draw_label(canvas, label_text, (offset_x, ly - 10), drawn_boxes)

    return canvas


def render_all_poses(
    test_name: str,
    raw_data: dict,
    reports: list[PoseAngleReport],
    out_dir: Path,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
) -> None:
    """
    한 테스트 이미지의 모든 포즈에 대해 각도 시각화 이미지를 저장합니다.
    """
    poses_raw = raw_data.get("poses", [])
    n = len(reports)

    for i, (raw_pose, report) in enumerate(zip(poses_raw, reports)):
        img = build_angle_image(raw_pose, report, canvas_w, canvas_h)

        if n == 1:
            filename = f"{test_name}_angle_plus_vis.png"
        else:
            filename = f"{test_name}_pose{i}_angle_plus_vis.png"

        out_path = out_dir / filename
        cv2.imwrite(str(out_path), img)
        print(f"  -> Angle Vis   : {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 콘솔 출력
# ─────────────────────────────────────────────────────────────────────────────

_SECTION = {
    "knee":     "Knee               (Hip - Knee - Ankle)",
    "elbow":    "Elbow Flexion      (Shoulder - Elbow - Wrist)",
    "shoulder": "Shoulder Abduction (Elbow - Shoulder - Hip)",
    "hip":      "Hip Flexion        (Shoulder - Hip - Knee)",
    "ankle":    "Ankle Dorsiflexion (Knee - Ankle - Foot Index)",
}

def print_report(test_name: str, result: dict) -> None:
    print()
    print("=" * 65)
    print(f"  {test_name.upper()}  |  {result['source_file']}")
    print(f"  Image : {result['image_width']} x {result['image_height']}  |  "
          f"Poses: {result['num_poses_detected']}  |  "
          f"Visibility threshold: {result['visibility_threshold']}")
    print("=" * 65)
    for report in result["_reports"]:
        print(f"\n  -- POSE #{report.pose_index} --")
        for section_key, section_title in _SECTION.items():
            print(f"\n  {section_title}")
            for j in report.joints:
                if section_key in j.joint:
                    print(j.display())
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Windows cp949 콘솔 인코딩 방어
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n[Tro-Fit] Joint Angle Calculator PLUS - Test Run")
    print(f"  Project root : {_PROJECT_ROOT}")
    print(f"  Input dir    : {LM_JSON_DIR}")
    print(f"  Output dir   : {ANGLE_DIR}")

    # 출력 디렉토리 생성
    ANGLE_DIR.mkdir(parents=True, exist_ok=True)
    ANGLE_JSON_DIR.mkdir(parents=True, exist_ok=True)
    ANGLE_IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for test_name, json_path in TEST_FILES.items():

        # ── 입력 파일이 없으면 results/ 루트에서 폴백 ──────────────────
        if not json_path.exists():
            fallback = RESULTS_DIR / f"{test_name}_landmarks.json"
            if fallback.exists():
                json_path = fallback
                print(f"\n  [WARN] landmark_json/ 없음 -> fallback: {fallback.name}")
            else:
                print(f"\n  [SKIP] 파일 없음: {json_path}")
                continue

        result = analyze_file(json_path)
        all_results[test_name] = result
        print_report(test_name, result)

        # ── 개별 angle JSON 저장 ───────────────────────────────────────
        save_data = {k: v for k, v in result.items() if not k.startswith("_")}
        per_path  = ANGLE_JSON_DIR / f"{test_name}_angle_plus.json"
        with per_path.open("w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"  -> Angle JSON  : {per_path}")

        # ── 각도 시각화 이미지 저장 ────────────────────────────────────
        render_all_poses(
            test_name  = test_name,
            raw_data   = result["_raw"],
            reports    = result["_reports"],
            out_dir    = ANGLE_IMG_DIR,
        )

    # ── 통합 angle JSON 저장 ──────────────────────────────────────────
    combined = {
        name: {k: v for k, v in res.items() if not k.startswith("_")}
        for name, res in all_results.items()
    }
    all_path = ANGLE_JSON_DIR / "angle_all_plus.json"
    with all_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"\n  -> Combined JSON: {all_path}")
    print("\n[DONE] All angle results saved.\n")


if __name__ == "__main__":
    main()
