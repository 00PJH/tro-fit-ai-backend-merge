# Tro-Fit Vision AI — 코드 심층 비교 분석

> **분석 대상:** 기존 4개 파일(`landmarks.py`, `pictographic_generator.py`, `pose_test.py`, `angle_calculator_test_plus.py`) vs `landmark_exporter.py` (main branch 추가분)  
> **관점:** 유지보수성, 확장성, 설계 일관성, 알고리즘 효율성  
> **결론 표기 기준:** `채택 유지` = 현행 설계를 그대로 유지 / `채택 흡수` = exporter의 해당 요소를 기존 시스템에 통합 / `제거 권고` = 사용하지 않음

---

## 1. 아키텍처 개요: 두 시스템의 데이터 흐름

### 기존 4파일 시스템 (Your System)
```
이미지 파일들 (img_test/)
      ↓
pose_test.py  ← [RunningMode.IMAGE]
  ├─ draw_landmarks_on_image()         # OpenCV BGR 시각화
  ├─ extract_landmarks_json()          # SSOT(BlazePoseLandmark) 기반 dict 구조 JSON 생성
  └─ generate_pictographic_svg()       # pictographic_generator.py 호출
        ↓
  results/joint_33/landmark_json/*.json   ← Dict 구조 (이름=Key)
        ↓
angle_calculator_test.py / angle_calculator_test_plus.py
  ├─ analyze_file()                    # JSON 파싱
  ├─ compute_*_angles()                # 관절별 각도 계산
  ├─ build_angle_image()               # 검은 배경 시각화
  └─ render_all_poses()               # 최종 이미지 저장
```

### landmark_exporter.py (main branch)
```
비디오 파일 or 웹캠
      ↓
landmark_exporter.py  ← [RunningMode.VIDEO]
  ├─ create_landmarker()               # 모델 로딩
  ├─ draw_pose_overlay()               # 실시간 프리뷰
  ├─ result_to_frame_record()          # 프레임별 List 구조 JSON 생성
  └─ export_landmarks()               # 전체 영상 처리 루프
        ↓
  results/landmarks.json              ← List 구조 (인덱스 기반)
  (angle 계산 파이프라인 없음)
```

---

## 2. 핵심 설계 비교: 랜드마크 네이밍 시스템

### 2-A. `landmark_exporter.py` 방식 — ❌ 하드코딩 배열

```python
# landmark_exporter.py L30~62
LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    ...
    "right_foot_index",
]
```

**문제점 분석:**
- `LANDMARK_NAMES[index]` 는 O(1)이지만, **이 배열 자체가 `landmarks.py`와 완전히 중복**된다.
- 만약 MediaPipe가 새 랜드마크(예: index 33=neck)를 추가하면, **이 파일과 `landmarks.py` 두 곳을 동시에 수정**해야 한다 → DRY 위반.
- `index < len(LANDMARK_NAMES)` 경계 검사를 직접 작성해야 하므로 범위 오류 시 **`IndexError` 대신 `f"landmark_{index}"` 같은 묵음 실패**가 발생한다.

```python
# landmark_exporter.py L80~87: landmark_to_dict()
def landmark_to_dict(landmark: Any, index: int) -> dict[str, Any]:
    return {
        "id": index,
        "name": LANDMARK_NAMES[index] if index < len(LANDMARK_NAMES) else f"landmark_{index}",
        # ↑ index가 범위를 벗어나도 에러 없이 조용히 폴백 → 디버깅 어렵
        "x": float(landmark.x),
        ...
        "presence": _optional_float(landmark, "presence"),  # ← visibility와 별도로 presence까지 추출
    }
```

---

### 2-B. 기존 시스템 방식 — ✅ IntEnum SSOT

```python
# landmarks.py L34~67: BlazePoseLandmark IntEnum
class BlazePoseLandmark(IntEnum):
    NOSE             = 0
    LEFT_EYE_INNER   = 1
    ...
    RIGHT_FOOT_INDEX = 32

    def json_key(self) -> str:
        return self.name.lower()  # "LEFT_SHOULDER" → "left_shoulder" O(1) 변환
```

**장점 분석:**
- **정방향/역방향 O(1) 조회**: `BPL(11).json_key()` → `"left_shoulder"`, `BPL["LEFT_SHOULDER"]` → `11`
- **런타임 타입 안전**: 범위 밖 int로 생성 시 즉시 `ValueError` → 조용한 실패 없음
- **IDE 자동완성**: `BPL.LEFT_SHOULDER`처럼 오타 시 즉시 `AttributeError`
- **단일 수정 포인트**: `landmarks.py`만 수정하면 `pose_test`, `pictographic_generator`, `angle_calculator` 전부 자동 반영

```python
# pose_test.py L85~98: extract_landmarks_json() — SSOT 활용
for idx, landmark in enumerate(pose_landmarks):
    try:
        safe_name = BlazePoseLandmark(idx).json_key()  # O(1), 타입 안전
    except ValueError:
        safe_name = f"landmark_{idx}"                  # 명시적 예외 처리
    
    landmarks_dict[safe_name] = {
        "x":          round(float(landmark.x), 6),
        "y":          round(float(landmark.y), 6),
        "z":          round(float(landmark.z), 6),
        "visibility": round(float(landmark.visibility), 6),
        "pixel_x":    int(landmark.x * image_width),   # ← 픽셀 좌표 미리 계산 (시각화 레이어에 편리)
        "pixel_y":    int(landmark.y * image_height),
    }
```

> **→ 채택 유지: 기존 시스템의 `BlazePoseLandmark IntEnum` 방식**  
> exporter의 `LANDMARK_NAMES` 배열은 `landmarks.py`와 중복이므로 통합 시 제거한다.

---

## 3. JSON 스키마 구조 비교

### 3-A. `landmark_exporter.py` — List(배열) 구조

```json
{
  "frame_index": 0,
  "landmarks": [
    { "id": 11, "name": "left_shoulder", "x": 0.45, "y": 0.32, "z": -0.1,
      "visibility": 0.99, "presence": 0.99 }
  ],
  "world_landmarks": [  ← ✅ 핵심 장점: 3D 미터 단위 좌표 포함
    { "id": 11, "name": "left_shoulder", "x": -0.15, "y": -0.3, "z": 0.05, ... }
  ]
}
```

**장점:**
- `world_landmarks` 포함 → 실제 물리 공간 3D 좌표 → **ROM 계산 정확도 향상 가능**
- `presence` 필드 포함 → visibility와 다른 신뢰도 지표

**단점:**
- 랜드마크를 이름(Key)으로 바로 접근 불가 → `lm_list[11]`처럼 인덱스로만 접근
- `angle_calculator`에서 쓰려면 `{lm["name"]: lm for lm in frame["landmarks"]}` 변환 필요
- 프레임 단위 구조라 `pose_test.py`의 이미지 기반 스키마와 호환되지 않음

### 3-B. 기존 시스템 — Dict(이름 Key) 구조

```json
{
  "poses": [{
    "pose_index": 0,
    "landmarks": {
      "left_shoulder": { "x": 0.45, "y": 0.32, "z": -0.1,
                         "visibility": 0.99, "pixel_x": 288, "pixel_y": 204 }
    }
  }]
}
```

**장점:**
- `landmarks["left_shoulder"]` 직접 O(1) 접근
- `angle_calculator`에서 `get_lm(landmarks, BPL.LEFT_SHOULDER)` 한 줄로 처리
- `pixel_x`, `pixel_y` 사전 계산으로 시각화 레이어 코드 단순화

**단점:**
- `world_landmarks` 없음 → 정규화 좌표만 사용 (카메라 원근 왜곡 영향 존재)
- `presence` 없음

> **→ 채택 유지: 기존 Dict(이름 Key) 구조 유지.**  
> 단, `world_landmarks` 필드와 `timestamp_ms` 필드는 **채택 흡수** 대상으로, 기존 스키마에 추가 병합한다.

---

## 4. 핵심 함수 라인-바이-라인 비교

### 4-A. 랜드마크 시각화 함수

#### `draw_landmarks_on_image()` — pose_test.py L20~64
```python
def draw_landmarks_on_image(image, detection_result, visibility_threshold=0.2):
    annotated_image = np.copy(image)           # ✅ 원본 보존
    h, w, _ = annotated_image.shape

    for pose_landmarks in pose_landmarks_list:
        coords = {}
        for idx, landmark in enumerate(pose_landmarks):
            if landmark.visibility >= visibility_threshold:  # ✅ 가시성 필터 적용
                coords[idx] = (int(landmark.x * w), int(landmark.y * h))

        for start_idx, end_idx in POSE_CONNECTIONS:   # ✅ landmarks.py SSOT 사용
            if start_idx in coords and end_idx in coords:
                cv2.line(annotated_image, ...)
        
        for idx, pt in coords.items():
            if idx in LEFT_LANDMARKS:   # ✅ frozenset O(1) 조회
                color = (255, 217, 0)   # 좌/우 색상 분리 → 임상적 가독성 우수
            elif idx in RIGHT_LANDMARKS:
                color = (0, 138, 255)
```
**평가:** LEFT/RIGHT 색상 분리, SSOT 활용, 가시성 필터 모두 우수. 임상 환경에서 좌/우 식별이 중요하기 때문에 이 설계가 핵심이다.

---

#### `draw_pose_overlay()` — landmark_exporter.py L117~156
```python
def draw_pose_overlay(frame_bgr: Any, landmarks: list[Any]) -> None:
    for start_idx, end_idx in POSE_CONNECTIONS:   # ← 파일 내 하드코딩된 POSE_CONNECTIONS
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        cv2.line(frame_bgr, start_point, end_point, (255, 120, 0), 2)  # ← 단색, 좌/우 구분 없음

    for index, landmark in enumerate(landmarks):
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame_bgr, point, 5, (0, 230, 0), -1)               # ← 단색 원
        if index in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):  # ← 매직 넘버!
            cv2.putText(frame_bgr, str(index), ...)                     # 숫자 레이블
```
**문제:** `(0, 11, 12, ...)` 는 전형적인 매직 넘버 안티패턴. `BPL.NOSE, BPL.LEFT_SHOULDER, ...`처럼 쓰면 의도가 명확해진다. 좌/우 색상 분리도 없다.

---

### 4-B. `result_to_frame_record()` — ✅ exporter 고유 장점

```python
# landmark_exporter.py L159~181
def result_to_frame_record(
    result: vision.PoseLandmarkerResult,
    frame_index: int,
    timestamp_ms: int,          # ← 비디오 시간축 정보! 기존 시스템에 없음
    width: int,
    height: int,
) -> dict[str, Any]:
    pose_landmarks  = result.pose_landmarks[0]  if result.pose_landmarks  else []
    world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else []
    #                 ↑ world_landmarks 추출 — 기존 시스템이 누락한 핵심 데이터

    return {
        "frame_index": frame_index,    # ← 시간축 추적 가능
        "timestamp_ms": timestamp_ms,  # ← 동작 구간 분석 가능
        "detected": bool(pose_landmarks),
        "landmark_count": len(pose_landmarks),
        "landmarks": [...],            # ← List 구조 (단점)
        "world_landmarks": [...],      # ← ✅ 3D 실좌표, ROM 정밀도 향상
    }
```
**평가:** `timestamp_ms`와 `world_landmarks`는 기존 시스템이 반드시 흡수해야 하는 핵심 기능이다. 특히 동적 ROM 분석(스쿼트 구간 자동 감지, 최대/최소 각도 포착)에 필수적이다.

---

### 4-C. 각도 계산 엔진 비교

#### 기존 시스템 — `calculate_angle_3d()` (angle_calculator_test_plus.py L158~178)
```python
def calculate_angle_3d(
    point_a: LandmarkPoint,
    vertex:  LandmarkPoint,
    point_c: LandmarkPoint,
) -> float:
    a = np.array([point_a.x, point_a.y, point_a.z], dtype=np.float64)  # float64 명시
    v = np.array([vertex.x,  vertex.y,  vertex.z],  dtype=np.float64)
    c = np.array([point_c.x, point_c.y, point_c.z], dtype=np.float64)

    va, vc = a - v, c - v
    n_va, n_vc = np.linalg.norm(va), np.linalg.norm(vc)

    if n_va < 1e-9 or n_vc < 1e-9:   # ✅ 동일 좌표 방어 (수치 안정성)
        return 0.0

    cos_a = float(np.clip(np.dot(va, vc) / (n_va * n_vc), -1.0, 1.0))  # ✅ arccos 도메인 클리핑
    return math.degrees(math.acos(cos_a))
```
**평가:** 수치 안정성까지 고려한 production-grade 구현. `landmark_exporter.py`에는 이에 해당하는 코드가 전혀 없다.  
> **→ 채택 유지: `calculate_angle_3d()` 현행 그대로 유지.**

---

### 4-D. `_compute()` — 가시성 게이트 패턴

```python
# angle_calculator_test_plus.py L201~216
def _compute(
    joint_name: str,
    lm_a: LandmarkPoint, lm_v: LandmarkPoint, lm_c: LandmarkPoint,
    threshold: float = VISIBILITY_THRESHOLD,
) -> JointAngleResult:
    res = JointAngleResult(
        joint=joint_name, angle_deg=None, reliable=False,
        # ↑ reliable=False를 기본값으로 → 가시성 미달 시 명시적 표시
        point_a=lm_a.name, vertex=lm_v.name, point_c=lm_c.name,
        visibility_a=lm_a.visibility, visibility_v=lm_v.visibility, visibility_c=lm_c.visibility,
        # ↑ 3개 점의 visibility를 모두 저장 → 디버깅/임상 로깅에 필수
    )
    if min(lm_a.visibility, lm_v.visibility, lm_c.visibility) < threshold:
        return res   # ← 3점 중 가장 낮은 visibility 기준으로 필터링 (가장 엄격한 방식)
    res.reliable  = True
    res.angle_deg = calculate_angle_3d(lm_a, lm_v, lm_c)
    return res
```
**평가:** `landmark_exporter.py`에는 이에 해당하는 신뢰도 게이트 로직이 없다. 임상/피트니스 앱에서 신뢰 불가 각도를 UI에 표시하면 안 되므로 이 패턴은 필수적이다.

---

### 4-E. 관절 정의 확장성 비교

#### `landmark_exporter.py` — 확장 불가 구조
```python
# 각도 계산 코드 자체가 없음. 랜드마크 추출만 함.
```

#### 기존 시스템 — 관절 추가가 1함수 추가로 완결

```python
# angle_calculator_test_plus.py: 새 관절 추가 방식
def compute_hip_angles(landmarks, threshold=...) -> list[JointAngleResult]:
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
```
새 관절 추가 시: 함수 1개 작성 + `analyze_pose()`에 `extend` 1줄 추가. 끝.  
> **→ 채택 유지: 이 패턴 현행 그대로 유지. `landmark_exporter`에는 이에 해당하는 구조 없음.**

---

### 4-F. `export_landmarks()` — 비디오 루프 구조

```python
# landmark_exporter.py L183~245
def export_landmarks(source, output_path, model_path, max_frames, preview):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0          # ✅ FPS 폴백 처리

    with create_landmarker(model_path) as landmarker:  # ✅ context manager로 자원 안전 해제
        while cap.isOpened():
            ok, frame_bgr = cap.read()
            if not ok:
                break

            timestamp_ms = int(frame_index * (1000.0 / fps))  # ✅ 프레임 → 타임스탬프 변환

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            # ↑ VIDEO 모드: 타임스탬프가 단조증가(monotonically increasing)해야 함 — 올바름

            if preview:
                cv2.imshow("Tro-Fit Landmark Exporter", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break    # ✅ 프리뷰 중 q키로 중단 가능

    cap.release()            # ✅ 명시적 자원 해제
    # ↑ 기존 시스템에 이 구조가 없음 — 반드시 흡수해야 할 패턴
```

---

### 4-G. pictographic_generator.py — 독립적 우수 설계

```python
# pictographic_generator.py L83~93: _build_coords()
def _build_coords(landmarks, scale_x, scale_y, visibility_threshold):
    coords = {}
    for bpl in BlazePoseLandmark:               # ✅ enum 순회로 전체 랜드마크 처리
        data = landmarks.get(bpl.json_key())    # ✅ SSOT O(1) 조회
        if data is None:
            continue
        if data.get("visibility", 0) < visibility_threshold:
            continue
        coords[bpl] = (
            data["pixel_x"] * scale_x,          # ✅ 사전계산된 pixel_x 활용
            data["pixel_y"] * scale_y,
        )
    return coords
```

```python
# pictographic_generator.py L96~98: _line_path()
def _line_path(a, b) -> str:
    return f"M {a[0]:.2f},{a[1]:.2f} L {b[0]:.2f},{b[1]:.2f}"
    # ✅ f-string으로 SVG path 생성 — 간결하고 정확
```
**평가:** 이 파일은 landmark_exporter의 List 구조와 호환되지 않는다. `landmark_exporter`의 JSON을 그대로 넣으면 `pixel_x` 필드가 없어서 바로 깨진다.

---

## 5. 종합 평가표

| 항목 | 기존 4파일 시스템 | landmark_exporter.py |
|------|:---:|:---:|
| **SSOT 랜드마크 네이밍** | ✅ IntEnum SSOT | ❌ 하드코딩 배열 |
| **JSON 스키마 일관성** | ✅ Dict(이름 Key) | ❌ List(인덱스) |
| **pixel_x/y 사전계산** | ✅ | ❌ |
| **world_landmarks 추출** | ❌ | ✅ |
| **timestamp 추적** | ❌ | ✅ |
| **비디오/웹캠 지원** | ❌ | ✅ |
| **각도 계산 엔진** | ✅ 완전 구현 | ❌ 없음 |
| **visibility 신뢰도 게이트** | ✅ | ❌ |
| **좌/우 색상 분리** | ✅ | ❌ |
| **자원 관리(context manager)** | ✅ (pose_test) | ✅ |
| **SVG 픽토그래픽** | ✅ | ❌ |
| **확장성(관절 추가)** | ✅ 함수 1개 추가 | N/A |
| **CLI 인터페이스** | ❌ | ✅ argparse |

---

## 6. 흡수해야 할 포인트 (우선순위)

### 🔴 Priority 1: `world_landmarks` 추출 추가

기존 `pose_test.py`의 `extract_landmarks_json()`에 `world_landmarks` 딕셔너리 추가.  
정규화 좌표(normalized) 대비 world 좌표는 카메라 거리 영향이 제거된 실제 미터 단위이므로, ROM 각도 계산 정확도가 상승한다.

```python
# 기존 landmarks_dict 아래에 추가
world_landmarks_dict = {}
for idx, wlm in enumerate(pose_world_landmarks):
    ...
```

### 🔴 Priority 2: 비디오/웹캠 파이프라인 (`RunningMode.VIDEO`)

`landmark_exporter.py`의 `export_landmarks()` 루프 구조를 흡수해서, 기존 `angle_calculator`가 비디오 스트림에서도 동작하도록 확장해야 한다.  
이때 JSON 스키마는 **기존 Dict 구조를 유지**하고, `frame_index`와 `timestamp_ms`만 추가 필드로 붙인다.

### 🟡 Priority 3: `argparse` CLI

`landmark_exporter.py`의 `parse_args()` 패턴을 `pose_test.py`에도 적용해서, `--input`, `--output`, `--model`, `--preview` 옵션을 CLI로 제어 가능하게 만들면 재사용성이 극대화된다.

### 🟡 Priority 4: `presence` 필드 추가

`landmark_to_dict()`에서 뽑아내는 `presence` 필드는 `visibility`와 다른 신뢰도 지표다. 향후 임상 데이터 품질 검증에 활용 가능하므로 스키마에 포함할 가치가 있다.

### 🟢 채택 유지 (수정 금지)

- `BlazePoseLandmark IntEnum` ([landmarks.py](../../../workspace/trofit/src/vision_ai/media_pipe_test/landmarks.py)) — SSOT 설계 완벽, 절대 변경하지 말 것
- `_compute()` 가시성 게이트 패턴 — 신뢰도 필터링 핵심 로직
- `calculate_angle_3d()` 수치 안정성 처리 — production-grade 구현
- `pictographic_generator.py` 전체 — `landmark_exporter` JSON(List 구조)과 호환되게 하지 말 것

---

## 7. 최종 권고 아키텍처

```
입력 레이어
  ├── 이미지 배치  → pose_test.py (IMAGE 모드) — 현행 유지
  └── 비디오/웹캠 → video_exporter.py (VIDEO 모드) — landmark_exporter 흡수 + SSOT 통합

공통 스키마 (Dict 구조 유지)
  ├── normalized landmarks: { "left_shoulder": { x, y, z, visibility, pixel_x, pixel_y } }
  └── world landmarks:      { "left_shoulder": { x, y, z, visibility } }  ← 추가

분석 레이어
  └── angle_calculator_test_plus.py — 현행 유지, world_landmarks 입력 경로 추가

시각화 레이어
  ├── draw_landmarks_on_image()    — pose_test.py (현행 유지)
  ├── build_angle_image()          — angle_calculator (현행 유지)
  └── pictographic_generator.py   — 현행 유지
```

---

## 8. 팀 역할 분담 권고 (조영진 팀원 전달용)

### 배경

`landmark_exporter.py`는 main 브랜치에서 조영진 팀원이 작성한 코드다.  
기존 4파일 시스템(박준형)과 설계 방향이 달라, 통합 시 충돌 포인트가 있다.  
아래는 **누가 어떤 작업을 해야 하는지** 명확히 분리한 역할표다.

---

### 조영진 팀원이 수행해야 할 작업 (landmark_exporter.py 측 수정)

| 작업 | 내용 | 이유 |
|------|------|---------|
| **SSOT 통합** | `LANDMARK_NAMES` 배열 삭제 후 `from src.vision_ai.media_pipe_test.landmarks import BlazePoseLandmark` 임포트로 교체 | DRY 원칙 위반 해소, landmarks.py와 중복 제거 |
| **매직 넘버 제거** | `if index in (0, 11, 12, ...)` → `if lm in (BPL.NOSE, BPL.LEFT_SHOULDER, ...)` | 의도 불명확, 유지보수 위험 |
| **JSON 스키마 협의** | `result_to_frame_record()` 출력 구조를 List→Dict로 맞추거나, 변환 헬퍼 함수 제공 | 기존 angle_calculator와 호환성 확보 |
| **POSE_CONNECTIONS 하드코딩 제거** | 파일 내 `POSE_CONNECTIONS` 삭제 후 `landmarks.py`에서 임포트 | 이미 SSOT가 있는데 중복 정의 |

### 박준형(본인)이 수행해야 할 작업 (기존 4파일 측 확장)

| 작업 | 내용 | 이유 |
|------|------|---------|
| **`world_landmarks` 추출 추가** | `pose_test.py`의 `extract_landmarks_json()`에 world 좌표 딕셔너리 필드 추가 | ROM 각도 계산 정확도 핵심 |
| **`video_exporter.py` 신규 작성** | exporter 루프 구조 흡수 + 기존 SSOT/Dict 스키마 적용한 비디오 전용 파이프라인 | 동적 ROM 분석(웹캠/영상) 필수 |
| **`angle_calculator`에 world_landmarks 경로 추가** | `get_lm()`이 world_landmarks도 받을 수 있게 파라미터 확장 | world 좌표 기반 각도 계산 활성화 |

---

### 요약: 누가 하는 게 더 효율적인가?

> **기존 4파일 시스템의 설계 원칙(SSOT, Dict 스키마, 가시성 게이트)을 박준형 본인이 완전히 이해하고 있으므로, `world_landmarks` 통합과 `video_exporter.py` 신규 작성은 박준형이 직접 진행하는 게 훨씬 빠르고 안전하다.**

조영진 팀원에게는 이 문서를 전달하면서 **landmark_exporter.py 내부의 설계 개선 포인트(SSOT 통합, 매직 넘버 제거)** 만 요청하면 된다. 그러면 두 파이프라인이 하나의 설계 원칙 위에서 동작하게 된다.
