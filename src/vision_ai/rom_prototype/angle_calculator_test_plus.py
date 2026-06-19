"""
angle_calculator_test_plus.py ??愿??媛곷룄 怨꾩궛 諛??쒓컖??紐⑤뱢 (?뺤옣 踰꾩쟾)
==========================================================

[異붽???愿??
  - 怨좉???援닿끝 (Hip Flexion)
  - 諛쒕ぉ 諛곗륫援닿끝 (Ankle Dorsiflexion)
  - ?닿묠 ?몄쟾 (Shoulder Abduction)
  - ?붽퓞移?援닿끝 (Elbow Flexion)

[???援ъ“]
  results/
  ?쒋?? joint_33/
  ??  ?쒋?? landmark_json/   : pose_test.py 媛 ?앹꽦 (媛쒕퀎/?듯빀 landmark JSON + CSV)
  ??  ?붴?? joint_img/       : pose_test.py 媛 ?앹꽦 (愿??異붿텧 寃곌낵 ?대?吏)
  ?쒋?? pictographic/        : pose_test.py 媛 ?앹꽦 (SVG ?쏀넗洹몃옒??
  ?붴?? angle/
      ?쒋?? angle_json/
      ??  ?쒋?? test_1_angle.json      : ?대?吏蹂?媛곷룄 寃곌낵
      ??  ?쒋?? test_2_angle.json
      ??  ?쒋?? test_3_angle.json
      ??  ?붴?? angle_all.json         : ?듯빀 寃곌낵
      ?붴?? angle_img/
          ?쒋?? test_1_angle_vis.png   : 媛곷룄 ?쒓컖???대?吏 (寃? 諛곌꼍 + 怨④꺽 + ?몃? 媛곷룄 ?덉씠釉?
          ?쒋?? test_2_angle_vis.png
          ?붴?? test_3_angle_vis.png

[媛곷룄 ?뺤쓽]
  - 臾대쫷(Knee)               : Hip ??Knee ??Ankle
  - ?붽퓞移?Elbow Flexion)    : Shoulder ??Elbow ??Wrist
  - ?닿묠(Shoulder Abduction) : Elbow ??Shoulder ??Hip
  - 怨좉???Hip Flexion)      : Shoulder ??Hip ??Knee
  - 諛쒕ぉ(Ankle Dorsiflexion) : Knee ??Ankle ??Foot Index

[?쒓컖???대?吏 ?ㅽ럺]
  - 寃? 諛곌꼍 (Black canvas)
  - 醫뚯륫 ?쒕뱶留덊겕 : 二쇳솴????(BGR 0,140,255)
  - ?곗륫 ?쒕뱶留덊겕 : ?섎뒛????(BGR 255,200,0)
  - ?곌껐??       : ?곗깋 (255,255,255)
  - 媛곷룄 ?덉씠釉?  : ?몃???(0,255,255 BGR) ??媛?愿?덉뿉 ?쒖떆
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np

# ?? ?덈? 寃쎈줈 湲곕컲 ?꾨줈?앺듃 猷⑦듃 異붽? ?????????????????????????????????????
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

# ?????????????????????????????????????????????????????????????????????????????
# ?곸닔 諛?寃쎈줈 ?ㅼ젙
# ?????????????????????????????????????????????????????????????????????????????

VISIBILITY_THRESHOLD: float = 0.65     # ??誘몃쭔 visibility ???좊ː 遺덇?

# ?낅젰 ?곗씠?? pose_test.py 媛 ?앹꽦??landmark JSON ?꾩튂
#   results/joint_33/landmark_json/test_X_landmarks.json
RESULTS_DIR   = _THIS_DIR.parent / "img_test" / "results"
JOINT33_DIR   = RESULTS_DIR / "joint_33"
LM_JSON_DIR   = JOINT33_DIR / "landmark_json"   # landmark JSON ?꾩슜 ?대뜑
ANGLE_DIR      = RESULTS_DIR / "angle"
ANGLE_JSON_DIR = ANGLE_DIR / "angle_json"
ANGLE_IMG_DIR  = ANGLE_DIR / "angle_img"

TEST_FILES: dict[str, Path] = {
    "test_1": LM_JSON_DIR / "test_1_landmarks.json",
    "test_2": LM_JSON_DIR / "test_2_landmarks.json",
    "test_3": LM_JSON_DIR / "test_3_landmarks.json",
}

# ?쒓컖???대?吏 罹붾쾭???ш린 (?뺢퇋??醫뚰몴瑜????ш린濡??ㅼ???
CANVAS_W = 640
CANVAS_H = 640
PADDING  = 60   # ?멸낸 ?щ갚 (?뺢퇋??醫뚰몴媛 0~1 ?대?濡??щ갚?쇰줈 怨듦컙 ?뺣낫)

# ?됱긽 (BGR)
COLOR_LEFT_PT   = (0,   140, 255)   # 二쇳솴????醫뚯륫 愿??
COLOR_RIGHT_PT  = (255, 200,   0)   # ?섎뒛?????곗륫 愿??
COLOR_OTHER_PT  = (255, 255, 255)   # ?곗깋   ??肄???以묐┰ 愿??
COLOR_LINE      = (255, 255, 255)   # ?곗깋 ?곌껐??
COLOR_LABEL     = (  0, 255, 255)   # ?몃???媛곷룄 ?덉씠釉?
COLOR_LABEL_BG  = (  0,   0,   0)   # 寃? 諛곌꼍 (罹붾쾭???먯껜)

FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.5
FONT_THICKNESS  = 1
LABEL_PADDING   = 4   # ?덉씠釉??띿뒪??諛곌꼍 ?⑤뵫

# ?????????????????????????????????????????????????????????????????????????????
# ?곗씠??援ъ“
# ?????????????????????????????????????????????????????????????????????????????

class LandmarkPoint(NamedTuple):
    """?⑥씪 ?쒕뱶留덊겕??3D 醫뚰몴? ?좊ː??"""
    x: float
    y: float
    z: float
    visibility: float
    name: str


@dataclass
class JointAngleResult:
    """愿???섎굹??媛곷룄 怨꾩궛 寃곌낵."""
    joint:        str
    angle_deg:    float | None   # None = visibility 誘몃떖
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
    """??Pose???꾩껜 愿??媛곷룄 由ы룷??"""
    pose_index: int
    joints: list[JointAngleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pose_index": self.pose_index,
            "joints": [asdict(j) for j in self.joints],
        }


# ?????????????????????????????????????????????????????????????????????????????
# ?듭떖 ?섑븰 ?붿쭊
# ?????????????????????????????????????????????????????????????????????????????

def calculate_angle_3d(
    point_a: LandmarkPoint,
    vertex:  LandmarkPoint,
    point_c: LandmarkPoint,
) -> float:
    """
    3D 踰≫꽣 ?댁쟻?쇰줈 vertex瑜?瑗?쭞?먯쑝濡??섎뒗 A-Vertex-C ?ъ씠媛?????諛섑솚?⑸땲??
    ?섏튂 ?덉젙?? ?숈씪 醫뚰몴 諛⑹뼱(1e-9), arccos ?대━??-1,1].
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


# ?????????????????????????????????????????????????????????????????????????????
# ?쒕뱶留덊겕 ?뚯떛
# ?????????????????????????????????????????????????????????????????????????????

def _parse(raw: dict, name: str) -> LandmarkPoint:
    return LandmarkPoint(
        x=float(raw["x"]), y=float(raw["y"]), z=float(raw["z"]),
        visibility=float(raw.get("visibility", 1.0)), name=name,
    )

def get_lm(landmarks: dict[str, dict], lm: BPL) -> LandmarkPoint:
    """BlazePoseLandmark enum?쇰줈 ?쒕뱶留덊겕瑜?媛?몄샃?덈떎 (SSOT ?곕룞)."""
    key = lm.json_key()
    return _parse(landmarks[key], key)


# ?????????????????????????????????????????????????????????????????????????????
# 愿?덈퀎 媛곷룄 怨꾩궛
# ?????????????????????????????????????????????????????????????????????????????

def _compute(
    joint_name: str,
    lm_a: LandmarkPoint, lm_v: LandmarkPoint, lm_c: LandmarkPoint,
    threshold: float = VISIBILITY_THRESHOLD,
) -> JointAngleResult:
    """visibility 寃利???媛곷룄 怨꾩궛. 誘몃떖 ??reliable=False 諛섑솚."""
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
    """臾대쫷: Hip ??Knee ??Ankle (醫뚯슦)"""
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
    """?붽퓞移?援닿끝: Shoulder ??Elbow ??Wrist (醫뚯슦)"""
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
    """?닿묠 ?몄쟾/嫄곗긽: Elbow ??Shoulder ??Hip (醫뚯슦)"""
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
    """怨좉???援닿끝: Shoulder ??Hip ??Knee (醫뚯슦)"""
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
    """諛쒕ぉ 諛곗륫援닿끝: Knee ??Ankle ??Foot Index (醫뚯슦)"""
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
    """?⑥씪 ?ъ쫰 ?뺤뀛?덈━ -> PoseAngleReport."""
    lm = pose["landmarks"]
    report = PoseAngleReport(pose_index=pose["pose_index"])
    report.joints.extend(compute_knee_angles(lm))
    report.joints.extend(compute_elbow_angles(lm))
    report.joints.extend(compute_shoulder_angles(lm))
    report.joints.extend(compute_hip_angles(lm))
    report.joints.extend(compute_ankle_angles(lm))
    return report


def analyze_file(json_path: Path) -> dict:
    """JSON ?뚯씪 ?섎굹瑜?遺꾩꽍?섏뿬 援ъ“?붾맂 寃곌낵 ?뺤뀛?덈━ 諛섑솚."""
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
        "_reports":             reports,   # 肄섏넄/?쒓컖?붿슜 (??????쒖쇅)
        "_raw":                 data,      # ?쒓컖??醫뚰몴 李몄“??(??????쒖쇅)
    }


# ?????????????????????????????????????????????????????????????????????????????
# ?쒓컖?? 寃? 諛곌꼍 怨④꺽 ?대?吏 + ?몃???媛곷룄 ?덉씠釉?
# ?????????????????????????????????????????????????????????????????????????????

def _norm_to_pixel(nx: float, ny: float, w: int, h: int,
                   pad: int = PADDING) -> tuple[int, int]:
    """?뺢퇋??醫뚰몴(0~1) ???⑤뵫???곸슜??罹붾쾭???쎌? 醫뚰몴."""
    inner_w = w - 2 * pad
    inner_h = h - 2 * pad
    px = int(nx * inner_w) + pad
    py = int(ny * inner_h) + pad
    return (px, py)


def _draw_label(canvas: np.ndarray, text: str, pt: tuple[int, int], drawn_boxes: list[tuple[int, int, int, int]] | None = None) -> None:
    """
    ?몃? 湲??+ 諛섑닾紐?寃? 諛곌꼍 諛뺤뒪濡?媛곷룄 ?덉씠釉붿쓣 洹몃┰?덈떎.
    ?띿뒪?멸? 罹붾쾭??諛뽰쑝濡??섍?吏 ?딅룄濡??먮룞 ?대━?묓븯硫?
    drawn_boxes 由ъ뒪?멸? 二쇱뼱吏硫?湲곗〈 ?곸옄? 寃뱀튂吏 ?딄쾶 ?꾩튂瑜?議곗젙?⑸땲??
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
                
            # 寃뱀튂硫??꾨옒濡??대룞
            y += int(box_h * 0.8)

    # ?띿뒪??諛뺤뒪媛 罹붾쾭?ㅻ? 踰쀬뼱?섏? ?딅룄濡?理쒖쥌 蹂댁젙
    x = max(LABEL_PADDING, min(x, canvas.shape[1] - tw - LABEL_PADDING * 2))
    y = max(th + LABEL_PADDING, min(y, canvas.shape[0] - LABEL_PADDING))

    x1_final = x - LABEL_PADDING
    y1_final = y - th - LABEL_PADDING
    x2_final = x + tw + LABEL_PADDING
    y2_final = y + LABEL_PADDING

    if drawn_boxes is not None:
        drawn_boxes.append((x1_final, y1_final, x2_final, y2_final))

    # 寃? 諛곌꼍 ?ш컖??
    cv2.rectangle(
        canvas,
        (x1_final, y1_final),
        (x2_final, y2_final),
        (30, 30, 30), cv2.FILLED
    )
    # ?몃? ?띿뒪??
    cv2.putText(canvas, text, (x, y), FONT, FONT_SCALE, COLOR_LABEL, FONT_THICKNESS, cv2.LINE_AA)


def build_angle_image(
    raw_pose: dict,
    angle_report: PoseAngleReport,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
) -> np.ndarray:
    """
    ?⑥씪 ?ъ쫰???뺢퇋??醫뚰몴瑜??ъ슜?섏뿬 寃? 諛곌꼍 怨④꺽 + 媛곷룄 ?덉씠釉??대?吏瑜??앹꽦?⑸땲??
    """
    canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    landmarks = raw_pose["landmarks"]

    # ?? 1. 醫뚰몴 留?援ъ꽦 (visibility 0.2 ?댁긽留? ??????????????????????????
    coords: dict[int, tuple[int, int]] = {}
    for lm in BPL:
        key = lm.json_key()
        if key not in landmarks:
            continue
        d = landmarks[key]
        if d.get("visibility", 1.0) < 0.2:
            continue
        coords[int(lm)] = _norm_to_pixel(d["x"], d["y"], canvas_w, canvas_h)

    # ?? 2. BODY_CONNECTIONS 湲곕컲 ?곗깋 ?곌껐???????????????????????????????
    for start_lm, end_lm in BODY_CONNECTIONS:
        s, e = int(start_lm), int(end_lm)
        if s in coords and e in coords:
            cv2.line(canvas, coords[s], coords[e], COLOR_LINE, 2, lineType=cv2.LINE_AA)

    # ?? 3. 愿???ъ씤??洹몃━湲??????????????????????????????????????????????
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

        cv2.circle(canvas, pt, 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)  # ???뚮몢由?
        cv2.circle(canvas, pt, 4, inner_color,     -1, lineType=cv2.LINE_AA)  # 而щ윭 ?대?

    # ?? 4. 媛곷룄 ?덉씠釉??쒖떆 ???????????????????????????????????????????????
    # joint ?대쫫 ??vertex BPL 留ㅽ븨
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

        # ?덉씠釉??띿뒪?? "knee: 110.7deg" (湲고샇???꾩뒪???덉쟾 泥섎━)
        short_name = joint_result.joint.replace("_", " ")   # e.g. "left knee"
        label_text = f"{short_name}: {joint_result.angle_deg:.1f}deg"

        # 愿???꾩튂 湲곗? 醫???遺꾨━ ?ㅽ봽???ㅼ젙
        lx, ly = coords[idx]
        
        if "left" in joint_result.joint:
            # ?띿뒪???ш린瑜?誘몃━ 怨꾩궛?섏뿬 愿?덉쓽 ?쇱そ??諛곗튂?섎룄濡??대룞
            (tw, th), _ = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)
            offset_x = lx - tw - 12
        else:
            # ?곗륫 愿?덉? 愿?덉쓽 ?ㅻⅨ履쎌뿉 諛곗튂
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
    ???뚯뒪???대?吏??紐⑤뱺 ?ъ쫰?????媛곷룄 ?쒓컖???대?吏瑜???ν빀?덈떎.
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


# ?????????????????????????????????????????????????????????????????????????????
# 肄섏넄 異쒕젰
# ?????????????????????????????????????????????????????????????????????????????

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


# ?????????????????????????????????????????????????????????????????????????????
# 硫붿씤
# ?????????????????????????????????????????????????????????????????????????????

def main() -> None:
    # Windows cp949 肄섏넄 ?몄퐫??諛⑹뼱
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n[Tro-Fit] Joint Angle Calculator PLUS - Test Run")
    print(f"  Project root : {_PROJECT_ROOT}")
    print(f"  Input dir    : {LM_JSON_DIR}")
    print(f"  Output dir   : {ANGLE_DIR}")

    # 異쒕젰 ?붾젆?좊━ ?앹꽦
    ANGLE_DIR.mkdir(parents=True, exist_ok=True)
    ANGLE_JSON_DIR.mkdir(parents=True, exist_ok=True)
    ANGLE_IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for test_name, json_path in TEST_FILES.items():

        # ?? ?낅젰 ?뚯씪???놁쑝硫?results/ 猷⑦듃?먯꽌 ?대갚 ??????????????????
        if not json_path.exists():
            fallback = RESULTS_DIR / f"{test_name}_landmarks.json"
            if fallback.exists():
                json_path = fallback
                print(f"\n  [WARN] landmark_json/ ?놁쓬 -> fallback: {fallback.name}")
            else:
                print(f"\n  [SKIP] ?뚯씪 ?놁쓬: {json_path}")
                continue

        result = analyze_file(json_path)
        all_results[test_name] = result
        print_report(test_name, result)

        # ?? 媛쒕퀎 angle JSON ??????????????????????????????????????????
        save_data = {k: v for k, v in result.items() if not k.startswith("_")}
        per_path  = ANGLE_JSON_DIR / f"{test_name}_angle_plus.json"
        with per_path.open("w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"  -> Angle JSON  : {per_path}")

        # ?? 媛곷룄 ?쒓컖???대?吏 ???????????????????????????????????????
        render_all_poses(
            test_name  = test_name,
            raw_data   = result["_raw"],
            reports    = result["_reports"],
            out_dir    = ANGLE_IMG_DIR,
        )

    # ?? ?듯빀 angle JSON ?????????????????????????????????????????????
    combined = {
        name: {k: v for k, v in res.items() if not k.startswith("_")}
        for name, res in all_results.items()
    }
    all_path = ANGLE_JSON_DIR / "angle_all_plus.json"
    with all_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"\n  -> Combined JSON: {all_path}")
    print("\n[DONE] All angle results saved.\n")



# -----------------------------------------------------------------------------
# Backend diagnosis payload helpers
# -----------------------------------------------------------------------------

def _safe_visibility(value: Any) -> float:
    return 1.0 if value is None else float(value)


def frame_record_to_pose(frame_record: dict[str, Any], pose_index: int = 0) -> dict | None:
    """Convert landmark_exporter.py frame record into analyze_pose() input."""
    if not frame_record.get("detected"):
        return None

    landmarks = {}
    for item in frame_record.get("landmarks", []):
        name = item.get("name")
        if not name:
            continue
        landmarks[name] = {
            "x": float(item["x"]),
            "y": float(item["y"]),
            "z": float(item["z"]),
            "visibility": _safe_visibility(item.get("visibility")),
        }

    return {"pose_index": pose_index, "landmarks": landmarks}


def _rom_from_values(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def _midpoint(a: LandmarkPoint, b: LandmarkPoint, name: str) -> LandmarkPoint:
    return LandmarkPoint(
        x=(a.x + b.x) / 2,
        y=(a.y + b.y) / 2,
        z=(a.z + b.z) / 2,
        visibility=min(a.visibility, b.visibility),
        name=name,
    )


def _spine_angle(landmarks: dict[str, dict]) -> float | None:
    try:
        left_shoulder = get_lm(landmarks, BPL.LEFT_SHOULDER)
        right_shoulder = get_lm(landmarks, BPL.RIGHT_SHOULDER)
        left_hip = get_lm(landmarks, BPL.LEFT_HIP)
        right_hip = get_lm(landmarks, BPL.RIGHT_HIP)
        left_knee = get_lm(landmarks, BPL.LEFT_KNEE)
        right_knee = get_lm(landmarks, BPL.RIGHT_KNEE)
    except KeyError:
        return None

    shoulder_mid = _midpoint(left_shoulder, right_shoulder, "mid_shoulder")
    hip_mid = _midpoint(left_hip, right_hip, "mid_hip")
    knee_mid = _midpoint(left_knee, right_knee, "mid_knee")
    if min(shoulder_mid.visibility, hip_mid.visibility, knee_mid.visibility) < VISIBILITY_THRESHOLD:
        return None
    return calculate_angle_3d(shoulder_mid, hip_mid, knee_mid)


def calculate_export_rom_metrics(export_payload: dict[str, Any]) -> dict[str, float]:
    """Calculate ROM metrics from landmark_exporter.py JSON payload."""
    angles_by_joint: dict[str, list[float]] = {
        "left_knee": [],
        "right_knee": [],
        "left_shoulder": [],
        "right_shoulder": [],
        "spine_flexion": [],
    }

    for index, frame in enumerate(export_payload.get("frames", [])):
        pose = frame_record_to_pose(frame, pose_index=index)
        if pose is None:
            continue

        report = analyze_pose(pose)
        for joint in report.joints:
            if joint.reliable and joint.angle_deg is not None and joint.joint in angles_by_joint:
                angles_by_joint[joint.joint].append(float(joint.angle_deg))

        spine_angle = _spine_angle(pose["landmarks"])
        if spine_angle is not None:
            angles_by_joint["spine_flexion"].append(float(spine_angle))

    return {
        "knee_left_rom": round(_rom_from_values(angles_by_joint["left_knee"]), 2),
        "knee_right_rom": round(_rom_from_values(angles_by_joint["right_knee"]), 2),
        "shoulder_left_rom": round(_rom_from_values(angles_by_joint["left_shoulder"]), 2),
        "shoulder_right_rom": round(_rom_from_values(angles_by_joint["right_shoulder"]), 2),
        "spine_flexion_rom": round(_rom_from_values(angles_by_joint["spine_flexion"]), 2),
    }


def calculate_diagnosis_flags(metrics: dict[str, float]) -> dict[str, Any]:
    knee_asymmetry = abs(metrics["knee_left_rom"] - metrics["knee_right_rom"]) > 8.0
    shoulder_asymmetry = abs(metrics["shoulder_left_rom"] - metrics["shoulder_right_rom"]) > 8.0

    falls_score = 0.0
    if metrics["knee_left_rom"] < 75.0 or metrics["knee_right_rom"] < 75.0:
        falls_score += 4.5
    if knee_asymmetry:
        falls_score += 3.5
    if metrics["spine_flexion_rom"] < 30.0:
        falls_score += 2.0

    return {
        "knee_asymmetry_detected": knee_asymmetry,
        "shoulder_asymmetry_detected": shoulder_asymmetry,
        "falls_risk_score": round(min(falls_score, 10.0), 2),
    }


def build_diagnosis_payload(
    export_payload: dict[str, Any],
    user_id: str,
    output_path: str,
) -> dict[str, Any]:
    """Build POST /api/v1/diagnosis payload from landmark_exporter.py output."""
    metrics = calculate_export_rom_metrics(export_payload)
    diagnosis = calculate_diagnosis_flags(metrics)
    return {
        "user_id": user_id,
        "source": export_payload.get("source", "webcam"),
        "landmark_file_path": output_path,
        "summary": {
            "total_frames": int(export_payload.get("total_frames", 0)),
            "detected_frames": int(export_payload.get("detected_frames", 0)),
            "detection_rate": float(export_payload.get("detection_rate_percent", 0.0)),
        },
        "metrics": metrics,
        "diagnosis": diagnosis,
    }

if __name__ == "__main__":
    main()





