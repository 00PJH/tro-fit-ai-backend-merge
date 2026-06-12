# 🧠 Tro-Fit WBS 현황 분석 — 실리콘밸리 시니어 AI 엔지니어 관점

> **분석 일자**: 2026-06-09 (오늘)  
> **분석 대상**: WBS 이미지 3개 항목 + 현재 코드베이스 전수 검토

---

## 📍 WBS 이미지에서 현재 위치 파악

WBS 이미지의 세 줄은 아래와 같습니다:

| 단계 | 역할 | 작업 | 담당 | 시작 | 마감 |
|------|------|------|------|------|------|
| 1차 개발 | Vision AI | 스마트폰 카메라 기반 **3D 관절 좌표 추출** 구현 | 조영진 | 05/22 | **06/12** |
| 1차 개발 | Vision AI | 33개 관절 **ROM 및 비대칭 분석 알고리즘** 개발 | 박준형 | 05/22 | **06/19** |
| 1차 개발 | Vision AI | 낙상 위험도 예측 **AI 모델 (FALLS Score)** 개발 | 박준형 | 06/02 | **06/26** |

### 🔍 현재 코드 완성도 진단

| 항목 | 상태 | 비고 |
|------|------|------|
| `angle_calculator_test.py` | ✅ 완성 | 3D 각도 계산 및 검은 배경+각도 레이블 시각화 통합 |
| `pose_test.py` | ✅ 완성 | MediaPipe 기반 랜드마크 추출 및 저장 파이프라인 |
| `pictographic_generator.py` | ✅ 완성 | SVG 픽토그래픽 자동 생성 구현 |
| `mediapipe_webcam.py` | ✅ 완성 | 실시간 FPS 벤치마크 구현 |
| `angle_all.json` 등 | ✅ 존재 | 각도 연산 및 추출 통합 결과 확인됨 |
| **FALLS Score (고도화)** | ⚠️ 초안 수준 | 현재는 룰 기반 3개 조건뿐, ML 모델 아님 |
| **실제 스마트폰 카메라 연동** | ❌ 미구현 | 현재는 이미지/비디오 파일 테스트 위주 |
| **33개 관절 전체 ROM** | ⚠️ 부분 완성 | 무릎/어깨/팔꿈치 구현 완료, 고관절·발목 등 미구현 |

---

## 🎯 오늘 당장 해야 하는 작업 (우선순위 순)

### 🥇 Priority 1 — **33개 관절 ROM 알고리즘 완성** (마감: 06/19)

**왜 지금 당장인가?**  
이미지의 두 번째 항목이 6/19 마감이고, 현재 무릎·어깨·척추 3종만 구현된 상태입니다.  
3D 관절 좌표 추출(첫 번째 항목 06/12 마감)이 완료되려면  
ROM 측정 대상 관절이 먼저 정의되어야 합니다.

**구체적으로 추가해야 하는 관절 ROM:**

```
현재 구현됨 (3종):
  ✅ 무릎 굴곡 (Knee) — 좌/우
  ✅ 어깨 거상 (Shoulder) — 좌/우
  ✅ 팔꿈치 굴곡 (Elbow) — 좌/우

추가 필요 (노인 낙상 위험 평가 핵심 관절):
  ❌ 고관절 굴곡 (Hip Flexion) — 좌/우
  ❌ 발목 배측굴곡 (Ankle Dorsiflexion) — 좌/우
  ❌ 척추 굴곡 (Spine Flexion)
  ❌ 어깨 외전 (Shoulder Abduction) — 좌/우
  ❌ 경추 굴곡 (Neck Flexion) — 선택
```

### 🥈 Priority 2 — **FALLS Score 모델 고도화** (마감: 06/26)

**현재 상태**: 룰 기반 if-else 3가지 조건 → 점수 합산 (매우 단순)  
**목표 상태**: 임상 데이터 기반의 의미 있는 낙상 위험 예측 모델

**실리콘밸리 관점의 핵심 판단:**  
> 지금 ML 모델을 새로 학습하려 하지 마세요. 데이터가 없습니다.  
> 대신 **임상 검증된 룰 기반 스코어링 시스템**을 정확하게 구현하는 것이 훨씬 현명합니다.

---

## 🛠️ 사전 세팅 / 사전 테스트 (반드시 먼저 할 것들)

### Step 0. 환경 검증 (30분, 오늘 당장)

```powershell
# 현재 venv 정상 작동 확인
.\venv\Scripts\python.exe -c "import mediapipe, cv2, numpy; print('OK')"

# angle_calculator_test.py 실제 실행 (각도 및 시각화 테스트)
.\venv\Scripts\python.exe src\vision_ai\rom_prototype\angle_calculator_test.py

# 실제 웹캠 연동 확인
.\venv\Scripts\python.exe src\vision_ai\benchmark\mediapipe_webcam.py
```

> [!IMPORTANT]
> JSON 결과 파일들이 생성된다는 것은 한 번은 파이프라인이 실행되었다는 뜻입니다.
> 하지만 **실제 목표 타겟인 어르신들의 동작 영상으로 테스트했는지** 반드시 확인하세요.

### Step 1. 기준 ROM 임상값 확보 (연구 1~2시간)

구현 전에 **각 관절의 정상 범위(Normal ROM)** 를 확보해야 합니다.  
룰 기반이든 ML이든, 이 기준값이 없으면 스코어링이 불가능합니다.

| 관절 | 정상 ROM (성인) | 노인 기준 (65세 이상) |
|------|----------------|---------------------|
| 무릎 굴곡 | 130~150° | 120° 이하면 주의 |
| 고관절 굴곡 | 110~130° | 90° 이하면 주의 |
| 발목 배측굴곡 | 15~20° | 10° 이하면 낙상 위험 ↑ |
| 어깨 굴곡 | 160~180° | 120° 이하면 주의 |
| 척추 굴곡 | 60~80° | 30° 이하면 주의 |

**참고 자료 (무료, 즉시 활용 가능):**
- [Berg Balance Scale (BBS)](https://pubmed.ncbi.nlm.nih.gov/) — 낙상 위험 임상 기준
- [Timed Up and Go (TUG) Test 기준](https://www.cdc.gov/steadi/index.html) — CDC STEADI 프로그램
- AROM 정상값: Norkin & White, "Measurement of Joint Motion: A Guide to Goniometry" (표준 교재)

### Step 2. 실제 동영상 테스트 데이터 확보

**현재 가장 큰 공백**: 실제 사람이 동작하는 영상으로 테스트한 적이 없습니다.

```
필요한 테스트 영상:
  1. 정면 카메라 - 스쿼트 동작 (무릎 굴곡 테스트용)
  2. 측면 카메라 - 팔 올리기 (어깨 굴곡 테스트용)
  3. 측면 카메라 - 앞으로 허리 굽히기 (척추 굴곡 테스트용)

촬영 조건:
  - 배경: 단색 (흰 벽 등) — MediaPipe 검출률 향상
  - 조명: 밝은 실내 (역광 금지)
  - 거리: 카메라에서 2~3m
  - 형식: .mp4, 720p 이상
```

---

## 📚 지금 당장 필요한 자료

### 🔬 기술 자료

| 자료 | 용도 | 링크/출처 |
|------|------|-----------|
| MediaPipe Pose Landmark 33개 인덱스 맵 | 나머지 관절 구현 | [공식 문서](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) |
| FALLS Score / BBS 임상 기준값 | FALLS Score 고도화 | CDC STEADI, PubMed |
| Tro-Fit PRD v1.0 (`doc/Tro-Fit_PRD_v1.0.md`) | 측정할 동작 5~7개 정의 확인 | 이미 있음 ✅ |

### 📹 테스트 데이터 (직접 확보 필요)

```
가장 빠른 방법:
  1. 본인이 직접 스마트폰으로 동작 영상 촬영 (5~10분)
  2. YouTube에서 "knee squat side view" 공개 영상 활용 (저작권 확인)
  3. AI 생성 영상은 MediaPipe가 잘 못 잡음 → 실사 영상 필수
```

---

## 🏗️ 핵심 질문: 코드를 프론트/백엔드 분리해야 하는가?

### 실리콘밸리 엔지니어의 결론: **YES, 반드시 분리해야 합니다**

이유는 단순합니다. Vision AI 코드는 **두 개의 서로 다른 런타임**에서 실행됩니다:

```
[스마트폰 앱 — React Native]
  └── On-device MediaPipe (TFLite)
      → 각도 계산은 JS/TS 로직
      → 결과: ROM 수치 (JSON)

[FastAPI 서버]
  └── Python MediaPipe (full model)
      → 서버사이드 분석 (고정밀)
      → 결과: FALLS Score + 안무 추천 (JSON)
```

### 분리 전략 상세

#### 🟢 지금 구현된 Python 코드 (순수 알고리즘층)

```text
angle_calculator_test.py   ← 3D 각도 수학 연산 및 시각화 통합
media_pipe_test/           ← MediaPipe 랜드마크 추출 및 픽토그래픽 생성 로직
```

#### 📱 프론트엔드(React Native)에 줘야 하는 형태

**React Native에서는 `@mediapipe/tasks-vision` 웹 버전 or TFLite 모델을 사용하므로,**  
각도 계산 로직을 TypeScript로 재구현해야 합니다.

```typescript
// angle_calculator.ts — Python angle_calculator.py의 TS 포트
export function calculateAngle3D(
  pointA: [number, number, number],
  pointB: [number, number, number],
  pointC: [number, number, number]
): number {
  const ba = pointA.map((v, i) => v - pointB[i]);
  const bc = pointC.map((v, i) => v - pointB[i]);
  const dotProduct = ba.reduce((sum, v, i) => sum + v * bc[i], 0);
  const normBA = Math.sqrt(ba.reduce((sum, v) => sum + v * v, 0));
  const normBC = Math.sqrt(bc.reduce((sum, v) => sum + v * v, 0));
  if (normBA < 1e-9 || normBC < 1e-9) return 0;
  const cosAngle = Math.max(-1, Math.min(1, dotProduct / (normBA * normBC)));
  return (Math.acos(cosAngle) * 180) / Math.PI;
}
```

#### 🖥️ 백엔드(FastAPI)에 줘야 하는 형태

Python 코드 그대로 사용하되, **FastAPI 엔드포인트로 래핑**:

```python
# POST /api/v1/analyze/rom
# Body: { "landmarks": [...], "session_id": "..." }
# Response: rom_analysis_result.json 형태

@router.post("/analyze/rom")
async def analyze_rom(request: RomAnalysisRequest) -> RomAnalysisResponse:
    landmarks_sequence = [deserialize_landmarks(f) for f in request.frames]
    return run_rom_pipeline_core(landmarks_sequence)
```

### 📦 최종 권장 패키지 구조

```
vision_ai/
├── core/                          ← 공통 알고리즘 (언어 무관 재구현 대상)
│   ├── angle_calculator.py        ← Python (백엔드)
│   └── angle_calculator.ts        ← TypeScript (프론트엔드)  ← 새로 만들어야 함
│
├── server/                        ← 백엔드 전용
│   ├── rom_service.py             ← FastAPI 서비스 레이어
│   └── falls_score_service.py     ← FALLS Score 서비스
│
└── mobile/                        ← 프론트엔드 전용
    ├── rom_calculator.ts           ← 온디바이스 ROM 계산
    └── landmark_utils.ts          ← MediaPipe 결과 파싱
```

---

## 🚦 오늘부터의 실행 순서 (Day-by-Day)

```
오늘 (6/9):
  1. angle_calculator_test.py 및 pose_test.py 실제 비디오로 파이프라인 확인 (웹캠 or 스마트폰 영상)
  2. 고관절·발목·척추 등 미구현 ROM 각도 함수 angle_calculator_test.py에 추가 구현
  3. 임상 FALLS Score 기준값 조사 (30분)

내일 (6/10~11):
  4. 나머지 관절 ROM 함수 전부 구현
  5. FALLS Score 고도화 (단순 if-else → 임상 가중치 기반 룰 엔진)
  6. TypeScript 포트 시작 (angle_calculator.ts)

6/12 이전:
  7. 실제 사람 영상으로 통합 테스트
  8. FastAPI 엔드포인트 뼈대 작성
  9. README + 결과 문서 업데이트
```

---

## ⚡ 실리콘밸리 전문가의 한마디

> **"Perfect is the enemy of good."**  
> 지금 FALLS Score를 ML 모델로 만들려고 시간 낭비하지 마세요.  
> **임상 검증된 룰 기반 스코어링**이 MVP에는 훨씬 더 강력합니다.  
> 실제 데이터 100건이 쌓이면 그때 ML로 전환하면 됩니다.
>
> 당신의 코드는 이미 꽤 잘 짜여져 있습니다. (`angle_calculator_test.py`의 각도 수학 함수 등)  
> 지금 필요한 건 "나머지 관절 추가" + "실제 영상 테스트" 두 가지입니다.
