# Tro-Fit Vision AI — 파이프라인 아키텍처 및 코드 통합 심층 분석 가이드

> **문서 목적:** 조영진 팀원의 `landmark_exporter.py` 기반 원본 코드와 기존의 ROM 각도 연산 시스템을 통합하여 탄생한 `mediapipe_rom_webcam_pipeline`의 전체 아키텍처와 통합 로직을 아주 상세하게 해설합니다.
> **수신자:** 조영진 팀원 (DB 연동 및 파이프라인 후속 작업자)

---

## 1. 아키텍처 개요: 두 시스템의 통합 및 데이터 흐름

조영진 팀원님의 원본 코드는 **'단일 비디오에서 관절 좌표를 추출하고 시각화'**하는 데 매우 훌륭한 기반이 되었습니다. 저는 이를 바탕으로, 단순 좌표 추출을 넘어 **"의학적 기준에 따른 가동범위(ROM) 각도 계산 및 비율 분석"**을 자동화하는 통합 파이프라인을 구축했습니다.

### 통합된 파이프라인 데이터 흐름 (Data Flow)
```text
비디오 파일 입력 (mp4)
      ↓
[1단계: 추출 엔진 - landmark_extractor.py] ← 조영진 팀원 코드 흡수 및 확장
  ├─ frame_bgr 추출 (cv2.VideoCapture)
  ├─ MediaPipe 추론 (detect_for_video 또는 detect)
  └─ world_landmarks 포함 Dict 구조 반환
      ↓
[2단계: 수학 엔진 - angle_engine.py] ← 박준형 기존 시스템
  ├─ landmarks.py의 SSOT(단일 진실 공급원) 참조
  ├─ 3D 벡터 내적 (calculate_angle_3d) 수행
  └─ 가시성(Visibility) 기반 신뢰도(reliable) 필터링
      ↓
[3단계: 분석 및 판정 - snapshot_rom_pipeline.py]
  ├─ normal_rom.json (의학적 기준각도) 로드
  ├─ Neutral(중립) vs Max(최대 수축) 프레임 비교
  └─ ROM 차이값 및 정상 범위 대비 비율(%) 계산
      ↓
[최종: 출력 및 DB 연동]
  └─ rom_result.json (최종 통합 포맷) 생성 → 조영진 팀원이 DB로 Insert
```

### 파이프라인 구조의 목적과 이점
1. **결합도 최소화 (Decoupling):** 영상 처리부(OpenCV/MediaPipe)와 수학 연산부(벡터 각도), 그리고 기준 데이터(normal_rom.json)를 완전히 분리했습니다. DB 연동을 담당하는 조영진 팀원님은 복잡한 3D 연산부를 전혀 볼 필요 없이, 파이프라인이 뱉어내는 최종 `rom_result.json`만 파싱하면 됩니다.
2. **단일 진실 공급원 (SSOT) 확립:** 기존에 흩어져 있던 관절 인덱스를 `landmarks.py` 하나로 통일하여(IntEnum 사용), 향후 MediaPipe 모델이 업데이트되거나 관절 번호가 바뀌어도 이 파일 단 1개만 수정하면 전체 파이프라인에 자동 적용되도록 안정성을 극대화했습니다.

---

## 2. 코드 통합 방식: 무엇을, 왜, 어떻게 통합했는가?

### 2-A. 조영진 팀원 코드의 채택 흡수 포인트
조영진 팀원님의 코드에서 가장 핵심적인 가치를 지닌 부분은 `result_to_frame_record()` 함수의 **`world_landmarks` 추출 및 시간축(`timestamp_ms`) 추적 로직**이었습니다.

```python
# 조영진 팀원 원본 코드 (landmark_exporter.py)
def result_to_frame_record(...):
    pose_landmarks = result.pose_landmarks[0] if result.pose_landmarks else []
    world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else []
    ...
```

**통합 로직 및 이유:**
- 기존 제 코드(`pose_test.py`)는 2D 픽셀 기반의 `normalized_landmarks`만 사용했으나, 이는 카메라 원근 왜곡에 취약합니다.
- 영진님의 위 코드를 채택하여, **미터(m) 단위의 실제 3D 물리 공간 좌표인 `world_landmarks`를 파이프라인의 기본 입력으로 격상**시켰습니다. 이를 통해 어깨 외전, 무릎 굴곡 등 3차원 공간에서 이루어지는 관절 각도 계산의 정밀도를 획기적으로 끌어올렸습니다.

### 2-B. 하드코딩 배열의 IntEnum 방식 통폐합 (방식 변경)
영진님 코드에는 아래와 같이 랜드마크 배열이 하드코딩되어 있었습니다.

```python
# 조영진 팀원 원본 코드 (제거됨)
LANDMARK_NAMES = ["nose", "left_eye_inner", "left_eye", ...]
...
name = LANDMARK_NAMES[index] if index < len(LANDMARK_NAMES) else f"landmark_{index}"
```

**통합 로직 및 이유:**
- 이 방식을 폐기하고, 저의 `landmarks.py`의 `BlazePoseLandmark` (IntEnum) 방식을 파이프라인 전체의 표준으로 강제했습니다.
- **왜?** 인덱스 초과 에러(`IndexError`) 발생 시 조용히 넘어가는(Silent Failure) 문제를 방지하고, 어떤 함수에서든 `BlazePoseLandmark.LEFT_SHOULDER.json_key()`를 호출해 O(1) 시간복잡도로 안전하게 JSON Key(`"left_shoulder"`)를 확정하기 위함입니다. DB에 들어갈 Key 값의 통일성을 무조건적으로 보장하기 위한 조치입니다.

---

## 3. 디렉토리 구조 (Directory Architecture) 상세 분석

파이프라인의 디렉토리는 단순한 스크립트 모음이 아니라, 하나의 완전한 **엔터프라이즈급 모듈**처럼 동작하도록 세분화하여 나누었습니다.

```text
vision_ai/mediapipe_rom_webcam_pipeline/
├── src/
│   ├── main.py                     # [CLI 진입점] 인자 파싱, 비디오 Auto-Ingest
│   ├── core/                       # [핵심 엔진] 도메인 로직 (불변)
│   │   ├── landmarks.py            # 관절 Enum 매핑 및 SSOT
│   │   ├── landmark_extractor.py   # MediaPipe 프레임 추출기
│   │   ├── angle_engine.py         # 3D 각도 계산 수학 엔진
│   │   └── visualizer.py           # 각도 캔버스 생성 및 오버레이
│   └── pipeline/                   # [비즈니스 로직] 엔진을 조합한 실행기
│       ├── snapshot_rom_pipeline.py# 3장 스냅샷 추출 기반 ROM 측정
│       └── rom_pipeline.py         # 비디오 전체 프레임 기반 추적
├── normal_rom.json                 # [설정] 의학적 기준각도 및 측정 축 정의
└── videos/                         # 비디오 파일 저장 (자동으로 부위별 분류됨)
```

### 왜 이런 디렉토리 구조로 나누었는가?
1. **관심사의 분리 (Separation of Concerns):** `core/` 내부의 파일들은 철저히 수학적 연산과 데이터 추출만 담당하며, 상태(State)를 가지지 않습니다. 반면 `pipeline/` 폴더 안의 스크립트들은 `core`의 기능들을 가져다 조립하여 "비디오의 0.5초 부분을 분석해라"와 같은 비즈니스 로직을 수행합니다. 
2. **확장성:** 만약 조영진 팀원님이 DB 저장 로직을 추가한다면, `core`를 건드릴 필요 없이 파이프라인 최상단(`pipeline/snapshot_rom_pipeline.py` 하단)에 `import db_connector` 한 줄만 추가하여 `final_result` 객체를 넘겨주기만 하면 되도록 안전성을 확보했습니다.
