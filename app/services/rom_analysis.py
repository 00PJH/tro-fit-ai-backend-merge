import math
import time
from enum import IntEnum
from typing import Any, Dict, List, Optional
import numpy as np

# ==============================================================================
# BlazePose Landmarks Definition (SSOT)
# ==============================================================================
class BlazePoseLandmark(IntEnum):
    NOSE             = 0
    LEFT_EYE_INNER   = 1
    LEFT_EYE         = 2
    LEFT_EYE_OUTER   = 3
    RIGHT_EYE_INNER  = 4
    RIGHT_EYE        = 5
    RIGHT_EYE_OUTER  = 6
    LEFT_EAR         = 7
    RIGHT_EAR        = 8
    MOUTH_LEFT       = 9
    MOUTH_RIGHT      = 10
    LEFT_SHOULDER    = 11
    RIGHT_SHOULDER   = 12
    LEFT_ELBOW       = 13
    RIGHT_ELBOW      = 14
    LEFT_WRIST       = 15
    RIGHT_WRIST      = 16
    LEFT_PINKY       = 17
    RIGHT_PINKY      = 18
    LEFT_INDEX       = 19
    RIGHT_INDEX      = 20
    LEFT_THUMB       = 21
    RIGHT_THUMB      = 22
    LEFT_HIP         = 23
    RIGHT_HIP        = 24
    LEFT_KNEE        = 25
    RIGHT_KNEE       = 26
    LEFT_ANKLE       = 27
    RIGHT_ANKLE      = 28
    LEFT_HEEL        = 29
    RIGHT_HEEL       = 30
    LEFT_FOOT_INDEX  = 31
    RIGHT_FOOT_INDEX = 32

    def json_key(self) -> str:
        return self.name.lower()

BPL = BlazePoseLandmark

# ==============================================================================
# Reference Normal ROM values
# ==============================================================================
NORMAL_ROM = {
    "shoulder_flexion": {"normal_deg": 150, "landmarks": ["hip", "shoulder", "elbow"]},
    "shoulder_extension": {"normal_deg": 40, "landmarks": ["hip", "shoulder", "elbow"]},
    "shoulder_abduction": {"normal_deg": 150, "landmarks": ["hip", "shoulder", "elbow"]},
    "elbow_flexion": {"normal_deg": 150, "landmarks": ["shoulder", "elbow", "wrist"]},
    "elbow_extension": {"normal_deg": 0, "landmarks": ["shoulder", "elbow", "wrist"]},
    "knee_flexion": {"normal_deg": 150, "landmarks": ["hip", "knee", "ankle"]},
    "knee_extension": {"normal_deg": 0, "landmarks": ["hip", "knee", "ankle"]},
    "hip_flexion": {"normal_deg": 90, "landmarks": ["shoulder", "hip", "knee"]},
    "ankle_dorsiflexion": {"normal_deg": 20, "landmarks": ["knee", "ankle", "foot_index"]},
}

VISIBILITY_THRESHOLD = 0.65

# ==============================================================================
# Core Mathematical Calculations
# ==============================================================================
def calculate_angle_3d(a: dict, v: dict, c: dict) -> float:
    """Calculate the 3D angle between vector v->a and v->c."""
    pos_a = np.array([a["x"], a["y"], a["z"]], dtype=np.float64)
    pos_v = np.array([v["x"], v["y"], v["z"]], dtype=np.float64)
    pos_c = np.array([c["x"], c["y"], c["z"]], dtype=np.float64)

    va, vc = pos_a - pos_v, pos_c - pos_v
    n_va, n_vc = np.linalg.norm(va), np.linalg.norm(vc)

    if n_va < 1e-9 or n_vc < 1e-9:
        return 0.0

    cos_a = float(np.clip(np.dot(va, vc) / (n_va * n_vc), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


def _compute_joint_angle(
    joint_name: str,
    lm_a: dict,
    lm_v: dict,
    lm_c: dict,
    threshold: float = VISIBILITY_THRESHOLD,
) -> dict:
    """Compute the joint angle and check reliability using visibility threshold."""
    vis_a = lm_a.get("visibility", 1.0)
    vis_v = lm_v.get("visibility", 1.0)
    vis_c = lm_c.get("visibility", 1.0)

    reliable = min(vis_a, vis_v, vis_c) >= threshold
    angle = calculate_angle_3d(lm_a, lm_v, lm_c) if reliable else None

    return {
        "joint": joint_name,
        "angle_deg": angle,
        "reliable": reliable,
        "point_a": lm_a.get("name", ""),
        "vertex": lm_v.get("name", ""),
        "point_c": lm_c.get("name", ""),
        "visibility_a": vis_a,
        "visibility_v": vis_v,
        "visibility_c": vis_c,
    }


def compute_joint_angles_for_pose(pose: dict, threshold: float = VISIBILITY_THRESHOLD, use_world: bool = True) -> list:
    """Compute all joint angles for a single pose."""
    if isinstance(pose, dict):
        lm = pose["world_landmarks"] if (use_world and "world_landmarks" in pose) else pose["landmarks"]
    else:
        lm = pose.world_landmarks if (use_world and getattr(pose, "world_landmarks", None) is not None) else pose.landmarks
    
    results = []

    def get_point(enum_val: BPL):
        key = enum_val.json_key()
        pt = None
        if isinstance(lm, dict):
            pt = lm.get(key)
        else:
            pt = getattr(lm, key, None)

        if pt is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0, "name": key}

        if isinstance(pt, dict):
            # Ensure dict has x, y, z
            return {"x": pt["x"], "y": pt["y"], "z": pt["z"], "visibility": pt.get("visibility", 1.0), "name": key}
        # In case Pydantic object is passed directly instead of dict
        return {"x": pt.x, "y": pt.y, "z": pt.z, "visibility": getattr(pt, "visibility", 1.0), "name": key}

    # Shoulders
    for side, elbow, shoulder, hip in [
        ("left", BPL.LEFT_ELBOW, BPL.LEFT_SHOULDER, BPL.LEFT_HIP),
        ("right", BPL.RIGHT_ELBOW, BPL.RIGHT_SHOULDER, BPL.RIGHT_HIP),
    ]:
        results.append(_compute_joint_angle(f"{side}_shoulder", get_point(elbow), get_point(shoulder), get_point(hip), threshold))

    # Elbows
    for side, shoulder, elbow, wrist in [
        ("left", BPL.LEFT_SHOULDER, BPL.LEFT_ELBOW, BPL.LEFT_WRIST),
        ("right", BPL.RIGHT_SHOULDER, BPL.RIGHT_ELBOW, BPL.RIGHT_WRIST),
    ]:
        results.append(_compute_joint_angle(f"{side}_elbow", get_point(shoulder), get_point(elbow), get_point(wrist), threshold))

    # Knees
    for side, hip, knee, ankle in [
        ("left", BPL.LEFT_HIP, BPL.LEFT_KNEE, BPL.LEFT_ANKLE),
        ("right", BPL.RIGHT_HIP, BPL.RIGHT_KNEE, BPL.RIGHT_ANKLE),
    ]:
        results.append(_compute_joint_angle(f"{side}_knee", get_point(hip), get_point(knee), get_point(ankle), threshold))

    # Hips
    for side, shoulder, hip, knee in [
        ("left", BPL.LEFT_SHOULDER, BPL.LEFT_HIP, BPL.LEFT_KNEE),
        ("right", BPL.RIGHT_SHOULDER, BPL.RIGHT_HIP, BPL.RIGHT_KNEE),
    ]:
        results.append(_compute_joint_angle(f"{side}_hip", get_point(shoulder), get_point(hip), get_point(knee), threshold))

    # Ankles
    for side, knee, ankle, foot_index in [
        ("left", BPL.LEFT_KNEE, BPL.LEFT_ANKLE, BPL.LEFT_FOOT_INDEX),
        ("right", BPL.RIGHT_KNEE, BPL.RIGHT_ANKLE, BPL.RIGHT_FOOT_INDEX),
    ]:
        results.append(_compute_joint_angle(f"{side}_ankle", get_point(knee), get_point(ankle), get_point(foot_index), threshold))

    return results

# ==============================================================================
# Frame Selection & Pipeline Execution
# ==============================================================================
def _visibility_score(joint_results: list, target_joint: str) -> float:
    """Sum minimum visibility of target joint points."""
    score = 0.0
    for j in joint_results:
        if target_joint in j["joint"]:
            score += min(j["visibility_a"], j["visibility_v"], j["visibility_c"])
    return score


def _extremeness_score(joint_results: list, target_joint: str) -> float:
    """Compute extremeness score (higher score = smaller angle)."""
    score = 0.0
    for j in joint_results:
        if target_joint in j["joint"] and j["angle_deg"] is not None and j["angle_deg"] > 0:
            score += 1.0 / j["angle_deg"]
    return score


def select_best_max_frame(
    candidates: list,
    target_joint: str,
) -> tuple[dict, str]:
    """Select the best frame representing the maximum movement posture."""
    if len(candidates) == 1:
        return candidates[0], "only_one_candidate"

    valid_candidates = []
    for c in candidates:
        pose = c["poses"][0] if isinstance(c, dict) else c.poses[0]
        # Calculate angles to evaluate visibility and extremeness
        angles = compute_joint_angles_for_pose(pose, VISIBILITY_THRESHOLD, True)
        # Check if target joints are reliable in this frame
        target_reliable = any(target_joint in j["joint"] and j["reliable"] for j in angles)
        if target_reliable:
            valid_candidates.append((c, angles))

    if not valid_candidates:
        # Fallback to candidates with target joint angles even if unreliable
        for c in candidates:
            pose = c["poses"][0] if isinstance(c, dict) else c.poses[0]
            angles = compute_joint_angles_for_pose(pose, VISIBILITY_THRESHOLD, True)
            valid_candidates.append((c, angles))

    if len(valid_candidates) == 1:
        return valid_candidates[0][0], "only_one_valid_candidate"

    # Evaluate candidates by visibility and extremeness
    vis_scores = [_visibility_score(angles, target_joint) for _, angles in valid_candidates]
    vis_diff = abs(vis_scores[0] - vis_scores[1]) if len(vis_scores) > 1 else 0

    if vis_diff > 0.5:
        best_idx = int(np.argmax(vis_scores))
        return valid_candidates[best_idx][0], "higher_visibility"

    extreme_scores = [_extremeness_score(angles, target_joint) for _, angles in valid_candidates]
    best_idx = int(np.argmax(extreme_scores))
    return valid_candidates[best_idx][0], "more_extreme_pose"


def analyze_rom_from_frames(request: Any) -> dict:
    """
    Core function to select neutral & max frames and compute ROM values.
    """
    start_time = time.perf_counter()

    # 1. Filter out invalid frames
    valid_frames = [
        f for f in request.frames
        if f.detected and len(f.poses) > 0
    ]

    if not valid_frames:
        raise ValueError("측정 가능한 프레임 정보가 없습니다. (포즈 감지 실패)")

    # Sort frames chronologically
    valid_frames = sorted(valid_frames, key=lambda x: x.timestamp_ms)

    # 2. Select neutral frame (closest to 500ms or first valid)
    neutral_frame = min(valid_frames, key=lambda x: abs(x.timestamp_ms - 500))

    # 3. Select max candidates (after neutral frame + 500ms, or fallback to all after neutral)
    max_candidates = [
        f for f in valid_frames
        if f.timestamp_ms > neutral_frame.timestamp_ms + 500
    ]
    if not max_candidates:
        max_candidates = [
            f for f in valid_frames
            if f.timestamp_ms > neutral_frame.timestamp_ms
        ]
    if not max_candidates:
        max_candidates = valid_frames

    # 4. Select best max frame
    best_max_frame, reason = select_best_max_frame(max_candidates, request.joint)

    # 5. Calculate joint angles for neutral and best max frames
    neutral_angles = {
        j["joint"]: j
        for j in compute_joint_angles_for_pose(neutral_frame.poses[0], VISIBILITY_THRESHOLD, True)
    }
    max_angles = {
        j["joint"]: j
        for j in compute_joint_angles_for_pose(best_max_frame.poses[0], VISIBILITY_THRESHOLD, True)
    }

    # 6. Compute ROM results for joints matching the target joint type
    rom_results = {}
    all_joints = set(neutral_angles.keys()) | set(max_angles.keys())
    target_joints = [j for j in all_joints if request.joint in j]

    for joint in target_joints:
        n_data = neutral_angles.get(joint)
        m_data = max_angles.get(joint)

        n_angle = n_data["angle_deg"] if (n_data and n_data["reliable"]) else None
        m_angle = m_data["angle_deg"] if (m_data and m_data["reliable"]) else None

        if n_angle is not None and m_angle is not None:
            rom_val = round(abs(n_angle - m_angle), 2)
            rom_results[joint] = {
                "neutral_angle": round(n_angle, 2),
                "max_angle": round(m_angle, 2),
                "ama_neutral_angle": round(n_angle, 2),
                "ama_max_angle": round(m_angle, 2),
                "rom": rom_val,
                "reliable": True,
            }
        else:
            rom_results[joint] = {
                "neutral_angle": round(n_angle, 2) if n_angle is not None else None,
                "max_angle": round(m_angle, 2) if m_angle is not None else None,
                "ama_neutral_angle": round(n_angle, 2) if n_angle is not None else None,
                "ama_max_angle": round(m_angle, 2) if m_angle is not None else None,
                "rom": None,
                "reliable": False,
                "reason": "neutral_unreliable" if n_angle is None else "max_unreliable",
            }

    # 7. Mobility Analysis
    joint_key = f"{request.joint}_{request.movement}"
    normal_entry = NORMAL_ROM.get(joint_key, {})
    normal_deg = normal_entry.get("normal_deg", 150)

    mobility_analysis = []
    for joint, result in rom_results.items():
        if result["reliable"] and result["rom"] is not None:
            ratio = round((result["rom"] / normal_deg) * 100, 1) if normal_deg > 0 else 0.0
            
            if ratio >= 75:
                grade = "NORMAL"
            elif ratio >= 50:
                grade = "WARNING"
            else:
                grade = "CRITICAL"

            clinical_meaning = ""
            joint_lower = request.joint.lower()
            if "shoulder" in joint_lower:
                if grade == "NORMAL":
                    clinical_meaning = "어깨 위 동작 포함 일상 동작 대부분 가능"
                elif grade == "WARNING":
                    clinical_meaning = "어깨 가동범위 제한 감지. 경미한 일상 불편 가능"
                else:
                    clinical_meaning = "기본 위생 동작 불가. 즉각 재활 권고"
            elif "elbow" in joint_lower:
                if grade == "NORMAL":
                    clinical_meaning = "팔꿈치 관절 정상 범위. 대부분의 일상 동작 가능"
                elif grade == "WARNING":
                    clinical_meaning = "팔꿈치 가동성 제한 감지. 스트레칭 권고"
                else:
                    clinical_meaning = "식사, 세면 등 일상 도구 사용 제한. 재활 권고"
            elif "knee" in joint_lower:
                if grade == "NORMAL":
                    clinical_meaning = "무릎 관절 정상 범위. 일상 보행 및 계단 이용 가능"
                elif grade == "WARNING":
                    clinical_meaning = "무릎 가동성 제한 감지. 쪼그려 앉기 등 제한 가능"
                else:
                    clinical_meaning = "정상 보행 및 계단 오르내리기 불가. 즉각 재활 권고"
            elif "hip" in joint_lower:
                if grade == "NORMAL":
                    clinical_meaning = "고관절 정상 범위. 앉고 서기 및 보행 대부분 가능"
                elif grade == "WARNING":
                    clinical_meaning = "고관절 가동성 제한 감지. 깊이 앉기 등 다소 불편"
                else:
                    clinical_meaning = "보행 장애 및 서기 곤란. 즉각 재활 권고"
            elif "ankle" in joint_lower:
                if grade == "NORMAL":
                    clinical_meaning = "발목 관절 정상 범위. 보행 및 균형 유지 가능"
                elif grade == "WARNING":
                    clinical_meaning = "발목 가동성 제한 감지. 경사로 등 보행 시 제한"
                else:
                    clinical_meaning = "발끝 들기 불가 및 보행 불안정. 즉각 재활 권고"
            else:
                if grade == "NORMAL":
                    clinical_meaning = "관절 가동범위 정상. 대부분의 일상 동작 가능"
                elif grade == "WARNING":
                    clinical_meaning = "관절 가동범위 제한 감지. 스트레칭 권고"
                else:
                    clinical_meaning = "해당 관절 기본 기능 수행 불가. 즉각 재활 권고"

            side_val = "both"
            if joint.startswith("left_"):
                side_val = "left"
            elif joint.startswith("right_"):
                side_val = "right"

            mobility_analysis.append({
                "side": side_val,
                "measured_angle_deg": result["rom"],
                "mobility_score": {
                    "normal_deg": float(normal_deg),
                    "rom_ratio": ratio,
                    "grade": grade,
                    "clinical_meaning": clinical_meaning
                }
            })

    # 8. Evaluate confidence
    reliable_count = sum(1 for v in rom_results.values() if v["reliable"])
    if reliable_count == 0:
        raise ValueError("관절 위치 검출 신뢰도가 낮아 가동범위(ROM)를 계산할 수 없습니다.")
    confidence = "HIGH" if reliable_count >= 2 else "MEDIUM" if reliable_count >= 1 else "LOW"

    elapsed_sec = time.perf_counter() - start_time

    # 9. Format response metadata
    max_candidate_strings = [
        f"{f.timestamp_ms / 1000.0:.3f}s (frame #{f.frame_index})"
        for f in max_candidates
    ]

    inferred_side = request.side
    if not inferred_side:
        if len(mobility_analysis) > 1:
            inferred_side = "both"
        elif len(mobility_analysis) == 1:
            inferred_side = mobility_analysis[0]["side"]
        else:
            inferred_side = "both"

    return {
        "session_id": request.session_id,
        "joint": request.joint,
        "movement": request.movement,
        "side": inferred_side,
        "video_file": request.video_file or f"{request.joint}_{request.movement}_{inferred_side}.mp4",
        "video_info": request.video_info or {
            "fps": 24.0,
            "total_frames": len(request.frames),
            "width": 1920,
            "height": 1080,
            "duration_s": round(len(request.frames) / 24.0, 3) if len(request.frames) > 0 else 0.0
        },
        "measurement": {
            "neutral_captured_at": f"{neutral_frame.timestamp_ms / 1000.0:.3f}s (frame #{neutral_frame.frame_index})",
            "max_candidates": max_candidate_strings,
            "max_selected": f"{best_max_frame.timestamp_ms / 1000.0:.3f}s (frame #{best_max_frame.frame_index})",
            "selection_reason": reason,
            "use_world_landmarks": True,
            "visibility_threshold": VISIBILITY_THRESHOLD,
        },
        "rom_results": rom_results,
        "mobility_analysis": mobility_analysis,
        "confidence": confidence,
        "elapsed_sec": round(elapsed_sec, 3),
        "model": request.model or "pose_landmarker_full.task"
    }
