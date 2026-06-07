import os
import glob
import json
import cv2
import numpy as np
import csv
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pictographic_generator import generate_pictographic_svg
from landmarks import BlazePoseLandmark, POSE_CONNECTIONS, LEFT_LANDMARKS, RIGHT_LANDMARKS




# ──────────────────────────────────────────────────────────────────────────────
# 1. 원본 이미지 위에 랜드마크 시각화
# ──────────────────────────────────────────────────────────────────────────────
def draw_landmarks_on_image(image, detection_result, visibility_threshold=0.2):
    """
    OpenCV BGR 이미지 위에 포즈 랜드마크와 연결선을 고품질로 직접 렌더링합니다.
    안티앨리어싱(LINE_AA)을 적용해 매끄러운 선과 흰색 테두리가 있는 원으로 그립니다.
    원의 크기를 약 1/3 수준으로 축소하여 콤팩트하게 표현합니다.
    """
    annotated_image = np.copy(image)
    h, w, _ = annotated_image.shape

    pose_landmarks_list = detection_result.pose_landmarks
    if not pose_landmarks_list:
        return annotated_image

    for pose_landmarks in pose_landmarks_list:
        coords = {}
        # 1. 랜드마크의 픽셀 좌표 구하기 (가시성 임계값을 넘는 것만)
        for idx, landmark in enumerate(pose_landmarks):
            if landmark.visibility >= visibility_threshold:
                px = int(landmark.x * w)
                py = int(landmark.y * h)
                coords[idx] = (px, py)

        # 2. 뼈대 연결선 그리기 (굵은 흰색 선, 안티앨리어싱 적용, 두께 2px)
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx in coords and end_idx in coords:
                cv2.line(annotated_image, coords[start_idx], coords[end_idx],
                         (255, 255, 255), 2, lineType=cv2.LINE_AA)

        # 3. 관절 포인트 그리기 (테두리가 있는 원, 안티앨리어싱 적용)
        for idx, pt in coords.items():
            # 바깥쪽 원 (흰색, 반지름 3)
            cv2.circle(annotated_image, pt, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

            # 안쪽 원 색상 결정 (좌: 하늘색, 우: 주황색, 기타: 흰색)
            if idx in LEFT_LANDMARKS:
                color = (255, 217, 0)   # BGR 하늘색
            elif idx in RIGHT_LANDMARKS:
                color = (0, 138, 255)   # BGR 주황색
            else:
                color = (255, 255, 255) # 코 등은 흰색

            # 안쪽 원 그리기 (반지름 2)
            cv2.circle(annotated_image, pt, 2, color, -1, lineType=cv2.LINE_AA)

    return annotated_image


# ──────────────────────────────────────────────────────────────────────────────
# 2. 관절 좌표 JSON 직렬화 구조로 변환
# ──────────────────────────────────────────────────────────────────────────────
def extract_landmarks_json(pose_landmarks_list, image_width, image_height):
    """
    감지된 포즈(들)의 33개 랜드마크 좌표를 JSON 직렬화 가능한 구조로 변환합니다.

    BlazePoseLandmark(idx).json_key() 로 JSON 키를 O(1) 에 확정합니다.
    (기존 LANDMARK_NAMES[idx] 방식 대비 경계 검사와 replace(" ", "_")가 불필요)
    """
    poses_data = []

    if not pose_landmarks_list:
        return poses_data

    for pose_idx, pose_landmarks in enumerate(pose_landmarks_list):
        landmarks_dict = {}

        for idx, landmark in enumerate(pose_landmarks):
            try:
                safe_name = BlazePoseLandmark(idx).json_key()  # O(1) 변환
            except ValueError:
                safe_name = f"landmark_{idx}"                  # 33개 범위 밖 방어

            landmarks_dict[safe_name] = {
                "x":          round(float(landmark.x), 6),
                "y":          round(float(landmark.y), 6),
                "z":          round(float(landmark.z), 6),
                "visibility": round(float(landmark.visibility), 6),
                "pixel_x":    int(landmark.x * image_width),
                "pixel_y":    int(landmark.y * image_height),
            }

        poses_data.append({
            "pose_index": pose_idx,
            "landmarks":  landmarks_dict,
        })

    return poses_data


# ──────────────────────────────────────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # 경로 설정
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    model_path = os.path.join(project_root, "models", "pose_landmarker_full.task")
    img_dir    = os.path.join(project_root, "src", "vision_ai", "img_test")
    output_dir = os.path.join(img_dir, "results")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Model Path     : {model_path}")
    print(f"Input Directory: {img_dir}")
    print(f"Output Directory: {output_dir}")

    # ── PoseLandmarker 설정 ──
    BaseOptions           = mp.tasks.BaseOptions
    PoseLandmarker        = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=3,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3
    )

    # 이미지 확장자 목록
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.BMP", "*.JPG", "*.JPEG", "*.PNG"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(img_dir, ext)))

    # 중복 제거 (대소문자 구분 없는 윈도우 환경 대응)
    image_paths = sorted(list(set(os.path.abspath(p) for p in image_paths)))

    if not image_paths:
        print("No images found in the input directory!")
        return

    print(f"Found {len(image_paths)} images to process.")

    # ── 출력 디렉토리 구조 생성 ──────────────────────────────────────────
    # results/joint_33/
    #   ├── landmark_json/ : 개별/통합 landmark JSON + CSV
    #   └── joint_img/     : 관절 추출 결과 이미지
    # results/pictographic/: 픽토그래픽 SVG 이미지
    joint33_dir   = os.path.join(output_dir, "joint_33")
    lm_json_dir   = os.path.join(joint33_dir, "landmark_json")
    joint_img_dir = os.path.join(joint33_dir, "joint_img")
    picto_dir     = os.path.join(output_dir, "pictographic")
    os.makedirs(lm_json_dir,   exist_ok=True)
    os.makedirs(joint_img_dir, exist_ok=True)
    os.makedirs(picto_dir,     exist_ok=True)

    # 벤치마크 결과 보관용 리스트
    benchmark_data = []
    # 전체 JSON 결과 (이미지별 33개 관절 좌표)
    all_landmarks_json = {}

    # ── PoseLandmarker 인스턴스 생성 및 실행 ──
    with PoseLandmarker.create_from_options(options) as landmarker:
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            print(f"\nProcessing: {filename}...")

            # OpenCV BGR 이미지 로드
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                print(f"  Failed to read image: {img_path}")
                continue

            h, w, _ = cv_image.shape

            # MediaPipe는 RGB 이미지를 요구하므로 변환
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # 추론 시간 정밀 측정
            start_time             = time.perf_counter()
            pose_landmarker_result = landmarker.detect(mp_image)
            end_time               = time.perf_counter()
            inference_time_seconds = end_time - start_time

            # 감지된 포즈 수
            num_poses_detected = (
                len(pose_landmarker_result.pose_landmarks)
                if pose_landmarker_result.pose_landmarks else 0
            )

            print(f"  -> Inference time: {inference_time_seconds:.4f}s, "
                  f"Detected poses: {num_poses_detected}")

            # ── [기능 1] 33개 관절 좌표 JSON 추출 ────────────────────────
            poses_landmark_data = extract_landmarks_json(
                pose_landmarker_result.pose_landmarks, w, h
            )

            safe_filename = filename.replace(" ", "_")
            all_landmarks_json[safe_filename] = {
                "image_width":        w,
                "image_height":       h,
                "num_poses_detected": num_poses_detected,
                "poses":              poses_landmark_data,
            }

            # 관절 좌표 콘솔 출력 (첫 번째 포즈만 샘플)
            if poses_landmark_data:
                print(f"  -> [Landmarks] 첫 번째 포즈 33개 관절 좌표 (정규화값):")
                for name, data in poses_landmark_data[0]["landmarks"].items():
                    print(f"     {name:25s}: x={data['x']:.4f}, y={data['y']:.4f}, "
                          f"z={data['z']:.4f}, vis={data['visibility']:.4f}")

            # ── 개별 landmark JSON → results/joint_33/landmark_json/ 저장 ─
            base_name         = os.path.splitext(safe_filename)[0]
            per_img_json_path = os.path.join(lm_json_dir, f"{base_name}_landmarks.json")
            try:
                with open(per_img_json_path, mode="w", encoding="utf-8") as f:
                    json.dump(all_landmarks_json[safe_filename], f,
                              ensure_ascii=False, indent=2)
                print(f"  -> JSON saved  : {per_img_json_path}")
            except Exception as e:
                print(f"  Failed to write per-image JSON for {filename}: {e}")

            # ── 벤치마크 데이터 추가 ──────────────────────────────────────
            benchmark_data.append({
                "image_name":             filename,
                "width":                  w,
                "height":                 h,
                "inference_time_seconds": round(inference_time_seconds, 4),
                "num_poses_detected":     num_poses_detected,
                "hardware_delegate":      "CPU (TFLite XNNPACK)",
                "model_name":             "pose_landmarker_full.task"
            })

            # ── 관절 시각화 이미지 → results/joint_33/joint_img/ 저장 ─────
            if num_poses_detected > 0:
                annotated_image = draw_landmarks_on_image(
                    cv_image, pose_landmarker_result, visibility_threshold=0.2
                )
            else:
                print(f"  No pose landmarks detected in {filename}.")
                annotated_image = cv_image

            out_path = os.path.join(joint_img_dir, filename)
            cv2.imwrite(out_path, annotated_image)
            print(f"  -> Annotated   : {out_path}")

            # ── [기능 2] 픽토그래픽 SVG → results/pictographic/ 저장 ─────
            picto_filename = f"{base_name}_pictographic.svg"
            picto_path     = os.path.join(picto_dir, picto_filename)
            generate_pictographic_svg(
                poses_data=poses_landmark_data,
                image_width=w,
                image_height=h,
                output_path=picto_path,
                visibility_threshold=0.2,
            )
            print(f"  -> Pictographic: {picto_path}")

    # ── CSV → results/joint_33/landmark_json/ 저장 ───────────────────
    csv_path    = os.path.join(lm_json_dir, "benchmark_results.csv")
    csv_headers = [
        "image_name", "width", "height",
        "inference_time_seconds", "num_poses_detected",
        "hardware_delegate", "model_name"
    ]
    try:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(benchmark_data)
        print(f"\nCSV saved: {csv_path}")
    except Exception as e:
        print(f"Failed to write CSV: {e}")

    # ── 전체 통합 landmark JSON → results/joint_33/landmark_json/ 저장 ─
    json_path = os.path.join(lm_json_dir, "landmarks_all.json")
    try:
        with open(json_path, mode="w", encoding="utf-8") as f:
            json.dump(all_landmarks_json, f, ensure_ascii=False, indent=2)
        print(f"All landmarks JSON saved: {json_path}")
    except Exception as e:
        print(f"Failed to write JSON: {e}")


if __name__ == "__main__":
    main()
