# LFM2-8B-A1B Terminal Agent Training Guide

> **목표**: Nemotron-Terminal-Corpus 비코딩 서브셋으로 LFM2-8B-A1B를 터미널 에이전트로 학습
> **전략**: Full FT SFT → 성능 확인 → GDPO RLVR (KLD 없이, TRL v0.27.0 네이티브)
> **환경**: Vessl AI H100 SXM x1 (80GB VRAM, 15 vCPUs, 200GB RAM, $2.39/hr)
> **배포**: FP16 → GGUF Q8_0 → llama.cpp CPU 16GB RAM 노트북

---

## 📚 참고 자료

| 항목 | URL |
|------|-----|
| Nemotron-Terminal 논문 | https://arxiv.org/html/2602.21193v1 |
| 데이터셋 | https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus |
| 학습 모델 | https://huggingface.co/unsloth/LFM2-8B-A1B |
| GDPO 논문 | https://arxiv.org/html/2601.05242v1 |
| TRL v0.27.0 (GDPO 네이티브) | https://github.com/huggingface/trl/releases/tag/v0.27.0 |
| Unsloth FP8 RL 가이드 | https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning |

---

## 🏗️ 프레임워크 스택

```
┌─────────────────────────────────────┐
│     사용자 코드 (학습 스크립트)        │
├─────────────────────────────────────┤
│  TRL v0.27.0 (GDPOTrainer/SFTTrainer) │ ← 알고리즘 (GDPO 네이티브)
├─────────────────────────────────────┤
│  Unsloth (FastLanguageModel)        │ ← 최적화 (FP8, 커스텀 커널)
├─────────────────────────────────────┤
│  vLLM                               │ ← 추론 (RLVR rollout 생성)
├─────────────────────────────────────┤
│  DeepSpeed ZeRO-2                   │ ← Full FT OOM 방지
└─────────────────────────────────────┘
```

| Phase | 용도 | 스택 |
|-------|------|------|
| Full FT SFT | 포맷 + 능력 학습 | Unsloth + TRL SFTTrainer + DeepSpeed |
| GDPO RLVR | RL 강화 | Unsloth + TRL GDPOTrainer + vLLM |
| 후속 RLVR | 풀 스케일 | TRL + vLLM + FSDP (A100 8대) |

---

## 🏗️ 파이프라인 개요

```
Phase 1: Full FT SFT (H100 1대, FP8)
    → 비코딩 터미널 데이터로 포맷 + 능력 학습
    → 여기서 성능 충분하면 배포 가능
    ↓
Phase 2: GDPO RLVR (H100 1대, LoRA, FP8)
    → SFT 모델 위에 RL 강화 (TRL v0.27.0 네이티브 GDPO)
    → KLD 없이, 빠르게 검증
    ↓
Phase 3 (후속): 풀 RLVR (A100 8대)
    ↓
배포: FP16 → GGUF Q8_0 → llama.cpp CPU 추론
```

---

## 🚀 Quick Start (Vessl AI H100)

### Step 1: Vessl AI 워크스페이스 생성

1. https://cloud.vessl.ai 접속
2. **Create workspace** 클릭
3. **GPU Type**: `VESSL H100 SXM x1` 선택 ($2.39/hr)
4. **Container image**: `Managed` → `Torch 2.9.1 (CUDA 13.0.1, Python 3.11)` 선택
5. **Persistent volume**: `Add volume` 필수! (체크포인트/모델 저장용)
6. **Create** 클릭

### Step 2: SSH 접속 또는 JupyterLab 열기

```bash
# SSH 접속 (Vessl CLI 사용)
pip install vessl
vessl configure
vessl workspace ssh <workspace-name>

# 또는 웹 브라우저에서 JupyterLab 사용
```

### Step 3: 프로젝트 설정 + 학습 실행

```bash
# 프로젝트 클론 + 패키지 설치
cd /root
git clone https://github.com/gyunggyung/Liquid-CLI.git
cd Liquid-CLI
bash setup_vessl.sh

# 전체 파이프라인 실행 (tmux 권장)
tmux new -s train

# 1. 데이터 준비 (~5분)
python prepare_data.py --output_dir /root/data

# 2. Full FT SFT (~3-5시간)
python train_sft.py \
    --data_path /root/data/sft_data \
    --output_dir /root/outputs/sft \
    --fp8 \
    --wandb

# 3. 평가 (~5분)
python evaluate.py --model_path /root/outputs/sft/final

# 4. GDPO RLVR (~2-4시간) — SFT 결과가 부족할 때만
python train_gdpo.py \
    --model_path /root/outputs/sft/final \
    --data_path /root/data/sft_data \
    --output_dir /root/outputs/gdpo \
    --max_steps 500 \
    --merge \
    --wandb

# 5. 모델 변환 (~10분)
python export_model.py \
    --model_path /root/outputs/gdpo/merged \
    --output_dir /root/outputs/gguf \
    --quant q8_0
```

### Step 4: 변환된 모델 다운로드

```bash
# 로컬로 다운로드 (Vessl CLI)
vessl volume download <workspace-name> /root/outputs/gguf ./gguf_model

# 또는 HuggingFace Hub에 업로드
python export_model.py \
    --model_path /root/outputs/gdpo/merged \
    --push_to_hub \
    --hub_repo gyunggyung/LFM2-8B-Terminal
```

---

## 📁 프로젝트 파일 구조

```
Liquid-CLI/
├── LFM2_Terminal_RLVR_Training_Guide.md  ← 이 가이드
├── requirements.txt                      ← Python 의존성
├── setup_vessl.sh                        ← Vessl AI 초기 설정 자동화
├── ds_config.json                        ← DeepSpeed ZeRO-2 설정
├── prepare_data.py                       ← 데이터 필터링 (비코딩 서브셋)
├── train_sft.py                          ← Phase 1: Full FT SFT
├── train_gdpo.py                         ← Phase 2: GDPO RLVR (TRL v0.27.0)
├── evaluate.py                           ← 터미널 에이전트 평가
└── export_model.py                       ← GGUF 변환 + Hub 업로드
```

---

## 🔍 각 파일 상세

### `prepare_data.py`

Nemotron-Terminal-Corpus에서 비코딩 데이터만 추출:

- `dataset_adapters` (226k, Math/Code/SWE) → **전부 제외**
- `skill_based_*` (140k, 9개 도메인) → **5개 비코딩 도메인만 선별**

| 포함 도메인 | 제외 도메인 |
|-----------|-----------|
| file_operations | data_science |
| data_processing | scientific_computing |
| data_querying | debugging |
| dependency_management | software_engineering |
| security | |

필터링 방법: domain 필드 기반 + 코딩 키워드 기반 이중 필터

### `train_sft.py`

- **Full Fine-Tuning** (LoRA 없음) — 모든 8B 가중치 업데이트
- Unsloth `FastLanguageModel` + FP8 양자화
- TRL `SFTTrainer` + packing (짧은 시퀀스 묶음)
- DeepSpeed ZeRO-2 옵션 (`--deepspeed_config ds_config.json`)
- 체크포인트 재개 (`--resume`)

DeepSpeed 사용:
```bash
python train_sft.py \
    --data_path /root/data/sft_data \
    --deepspeed_config ds_config.json
```

### `train_gdpo.py`

- SFT 모델 위에 **LoRA** (r=64, all-linear) 적용
- **TRL v0.27.0의 네이티브 GDPO** 사용 (`use_gdpo=True`)
- vLLM으로 rollout 생성 (4 generations/prompt)
- 4종 독립 리워드 함수:
  - R1: JSON 포맷 준수 (analysis/plan/commands 구조)
  - R2: 명령어 품질 (유효한 Linux 명령어 사용)
  - R3: 추론 품질 (분석/계획의 의미 + 터미널 키워드)
  - R4: 응답 길이 제어

### `evaluate.py`

10개 비코딩 터미널 태스크로 평가:
- file_operations (2개)
- data_processing (2개)
- data_querying (1개)
- dependency_management (2개)
- security (1개)
- networking (1개)
- system_admin (1개)

평가 지표:
- JSON 포맷 준수율
- 관련 명령어 사용율
- 카테고리별 평균 점수

### `export_model.py`

학습된 모델 → GGUF 변환:
- `q8_0`: INT8, ~8GB (16GB RAM 노트북용)
- `q4_k_m`: 4bit, ~4.5GB (더 가벼운 환경용)
- HuggingFace Hub 업로드 옵션

---

## 📊 예상 시간표 및 비용

| 단계 | 시간 | 비용 ($2.39/hr) |
|------|------|----------------|
| 데이터 다운로드 + 필터링 | ~10분 | $0.4 |
| Phase 1: Full FT SFT (2 epochs) | 3~5시간 | $7~12 |
| 평가 | ~5분 | $0.2 |
| Phase 2: GDPO RLVR (500 steps) | 2~4시간 | $5~10 |
| GGUF 변환 | ~10분 | $0.4 |
| **총합** | **5.5~9.5시간** | **$13~23** |

---

## ⚠️ VRAM 예산 (H100 80GB)

### Phase 1: Full FT SFT
```
모델 (FP8):        ~8 GB
Optimizer (FP32):  ~32 GB (DeepSpeed 없이)
Gradients:         ~8 GB
Activations:       ~15 GB (gradient checkpointing)
───────────────────────────
합계:              ~63 GB ← 80GB 이내 ✅

DeepSpeed ZeRO-2 사용 시:
  Optimizer → CPU offload → ~44 GB ✅
```

### Phase 2: GDPO RLVR (LoRA)
```
모델 (FP8):        ~8 GB (vLLM과 공유)
vLLM KV Cache:     ~15 GB
LoRA (BF16):       ~2 GB
Optimizer (8bit):  ~4 GB
Activations:       ~15 GB
───────────────────────────
합계:              ~44 GB ← 80GB 이내 ✅✅
```

---

## 🔧 주요 커맨드라인 옵션

### train_sft.py

| 옵션 | 기본값 | 설명 |
|------|-------|------|
| `--model_name` | `unsloth/LFM2-8B-A1B` | 모델 |
| `--max_seq_length` | 8192 | 최대 시퀀스 길이 |
| `--batch_size` | 1 | per-device batch |
| `--grad_accum` | 8 | gradient accumulation |
| `--learning_rate` | 2e-5 | 학습률 |
| `--num_epochs` | 2 | 에폭 수 |
| `--max_steps` | -1 | 최대 스텝 (-1=전체) |
| `--fp8` | True | FP8 양자화 |
| `--deepspeed_config` | None | ds_config.json 경로 |
| `--resume` | False | 체크포인트에서 재개 |

### train_gdpo.py

| 옵션 | 기본값 | 설명 |
|------|-------|------|
| `--model_path` | SFT 결과 경로 | 학습할 모델 경로 |
| `--lora_rank` | 64 | LoRA rank |
| `--num_generations` | 4 | prompt당 생성 수 |
| `--max_steps` | 500 | 빠른 검증용 (A100에서 2000+) |
| `--kl_coef` | 0.05 | KL divergence 계수 |
| `--merge` | False | LoRA 머지 저장 |

---

## 🔧 트러블슈팅

| 문제 | 해결 |
|------|------|
| Full FT OOM | `--deepspeed_config ds_config.json` 추가, 또는 `--batch_size 1 --grad_accum 16` |
| SFT 후 JSON 포맷 미준수 | `--num_epochs 3` 또는 `--learning_rate 3e-5` |
| RLVR 보상 0 유지 | SFT가 불충분 → SFT epoch 늘리기 |
| vLLM OOM | `--num_generations 2` (4→2), 또는 `--max_seq_length 4096` |
| GGUF 변환 실패 | `--merge` 옵션으로 먼저 FP16으로 머지 후 변환 |
| Vessl AI 재시작 후 데이터 사라짐 | persistent volume 사용 + `/root/` 경로에 저장 |
| Unsloth 미지원 모델 | `--use_unsloth` 플래그 빼고 transformers 직접 사용 |

---

## 📝 후속 작업 (A100 8대)

```bash
# A100 8대 환경에서의 변경 사항:
# 1. DeepSpeed → FSDP 전환
# 2. num_generations: 4 → 8
# 3. max_steps: 500 → 2000+
# 4. KLD 추가 가능 (LFM2-24B-A2B ref_model)
# 5. max_seq_length 확대 (8192 → 16384)
# 6. 전체 데이터셋 사용 (코딩 도메인 포함)

torchrun --nproc_per_node=8 train_gdpo.py \
    --model_path /root/outputs/sft/final \
    --batch_size 4 \
    --num_generations 8 \
    --max_steps 2000
```
