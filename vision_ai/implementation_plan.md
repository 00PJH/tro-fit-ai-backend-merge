# 관절/동작별 폴더 구조 기반 ROM 파이프라인 리팩토링

## 배경

현재 결과물은 영상 파일명 기반으로 results/{video_stem}/ 에 단순 저장되어
동일 동작 반복 시 덮어쓰기, 관절/동작 구분 불가, 세션 추적 불가 등의 문제 존재.

## 목표 폴더 구조

```
mediapipe_rom_webcam_pipeline/
├── videos/
│   └── {joint}/            예: shoulder / knee / elbow
│       └── {movement}/     예: flexion / extension / abduction
│           └── *.mp4       사용자가 영상을 여기에 저장
│
└── results/
    └── {joint}/
        └── {movement}/
            └── {YYYYMMDD_HHMMSS}/   세션 타임스탬프 (덮어쓰기 방지)
                ├── input.mp4        원본 영상 복사본
                ├── snapshots/
                │   ├── frames/      관절 오버레이 이미지
                │   └── angle_vis/   골격 + 각도 레이블 이미지
                └── rom_result.json  ROM 수치 + 정상비율 포함
```

## 변경 대상 파일

---

### 신규 파일

#### [NEW] normal_rom.json
경로: mediapipe_rom_webcam_pipeline/normal_rom.json

AMA 기준 정상 ROM 수치 데이터.
각 joint/movement 조합에 대한 normal_deg 값 포함.

---

### 수정 파일

#### [MODIFY] main.py
- `--joint` 인수 추가 (선택적): shoulder / knee / elbow
- `--movement` 인수 추가 (선택적): flexion / extension / abduction
- `--video` 경로가 videos/{joint}/{movement}/ 계층 구조이면 joint/movement 자동 파싱
- output_dir 계산 로직 변경:
  - 기존: results/{video_stem}/
  - 변경: results/{joint}/{movement}/{YYYYMMDD_HHMMSS}/
- 처리 완료 후 input.mp4를 결과 폴더에 복사

#### [MODIFY] snapshot_rom_pipeline.py
- run_snapshot_rom_pipeline 반환 dict에 joint, movement, session_id 필드 추가
- rom_result.json 저장 시 정상 ROM 대비 비율(rom_ratio) 필드 추가
  - rom_ratio = (측정 ROM / 정상각도) * 100

---

## 정상 ROM 기준값 (normal_rom.json)

```json
{
  "shoulder": {
    "flexion":   {"normal_deg": 150, "view": "lateral"},
    "extension": {"normal_deg": 40,  "view": "lateral"},
    "abduction": {"normal_deg": 150, "view": "frontal"}
  },
  "elbow": {
    "flexion":   {"normal_deg": 150, "view": "lateral"},
    "extension": {"normal_deg": 0,   "view": "lateral"}
  },
  "knee": {
    "flexion":   {"normal_deg": 150, "view": "lateral"},
    "extension": {"normal_deg": 0,   "view": "lateral"}
  }
}
```

## Open Questions

> [!IMPORTANT]
> videos/ 폴더 계층이 없는 경우(예: --video로 임의 경로 지정 시) joint/movement 값이 unknown으로 처리됩니다.
> 이 경우 results/unknown/unknown/{timestamp}/ 에 저장하는 방식으로 구현 예정입니다.
> 별도 처리 방식이 필요하면 알려주세요.

> [!NOTE]
> 기존 results/ 폴더의 데이터(shoulder_test/ 등)는 새 구조와 무관하게 그대로 유지됩니다.
> 새 구조는 이후 실행분부터 적용됩니다.

## 검증 계획

1. shoulder/abduction/ 폴더에 shoulder_test.mp4 이동 후 실행
2. results/shoulder/abduction/{timestamp}/ 생성 확인
3. input.mp4 복사본 존재 확인
4. rom_result.json에 rom_ratio 필드 존재 확인
