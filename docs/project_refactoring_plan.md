# 프로젝트 구조화 및 리팩토링 계획 (Project Refactoring Plan)

본 문서는 **Liquid-CLI** 프로젝트의 파일 관리 효율성을 높이기 위해 제안된 구조화 방안입니다. 현재 학습이 진행 중이므로, 모든 프로세스가 종료된 후 적용하는 것을 권장합니다.

## 1. 폴더 구조 제안 (Target Structure)

```
Liquid-CLI/
├── docs/                 # 순수 문서 및 가이드 (수정 완료)
├── configs/              # [NEW] 설정 파일 모음
├── logs/                 # [NEW] 학습 로그 및 평가 결과
├── scripts/              # [NEW] 유틸리티 스크립트
├── archive/              # [NEW] 이전/실험용 코드 보관
├── train_sft.py          # 핵심 실행 파일 (루트 유지)
├── train_gdpo.py         # 핵심 실행 파일 (루트 유지)
├── README.md
├── requirements.txt
└── LICENSE
```

## 2. 세부 이동 대상 및 작업 내용

### A. 로그 및 결과물 관리 (`logs/`)
- **이동 대상**:
  - `docs/로스정리.txt` → `logs/train_loss.log`
  - `docs/오류.txt` → `logs/error.log`
  - `docs/eval_results.json` → `logs/eval_results.json`
- **변경 사항**: `evaluate.py` 및 `evaluate_vllm.py` 내의 결과 저장 경로(`output_dir`)를 `logs/`로 수정.

### B. 설정 파일 관리 (`configs/`)
- **이동 대상**:
  - `ds_config.json` → `configs/ds_config.json`
- **변경 사항**: `train_sft.py` 실행 시 `--deepspeed_config configs/ds_config.json`으로 인자 수정.

### C. 유틸리티 스크립트 관리 (`scripts/`)
- **이동 대상**:
  - `prepare_data.py` → `scripts/prepare_data.py`
  - `export_model.py` → `scripts/export_model.py`
  - `evaluate_vllm.py` → `scripts/evaluate_vllm.py`
- **변경 사항**: `README.md`의 실행 커맨드 가이드에서 경로 수정.

### D. 코드 보관 (`archive/`)
- **이동 대상**:
  - `train_unsloth.py`
  - `train_unsloth_processed.py`
- **이유**: 메인 파이프라인에서 벗어난 코드를 격리하여 가독성 확보.

## 3. 리팩토링 체크리스트 (종료 후)
- [ ] 폴더 생성: `mkdir logs configs archive`
- [ ] 파일 이동: `mv` 명령어로 대상 파일 이동
- [ ] 경로 수정: `README.md` 내 링크 및 커맨드 업데이트
- [ ] 스크립트 수정: `evaluate.py` 등 내부 저장 경로 변수 업데이트

---
*작성일: 2026-04-18*
*참고: 현재 학습 중에는 절대 경로를 변경하지 마십시오.*
