# LFM2 Terminal Agent 🖥️

> LiquidAI의 LFM2-8B-A1B를 터미널 에이전트로 학습시키는 프로젝트

## 🎯 목표

CPU 노트북(16GB RAM)에서 돌아가는 터미널 에이전트를 만든다.
NVIDIA의 [Nemotron-Terminal-Corpus](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus) 데이터로 학습하고, GGUF로 변환해서 llama.cpp로 실행한다.

## 💡 왜 이 모델인가?

- **LFM2-8B-A1B**: LNN(Liquid Neural Network) 아키텍처 — gated short conv + GQA(Grouped Query Attention) 하이브리드. 8B 파라미터지만 1B만 활성 → CPU에서 빠름
- **비코딩 집중**: 파일 관리, 데이터 처리, 보안, 패키지 관리 등 일상적 터미널 작업에 특화
- **GDPO**: 다중 리워드 강화학습으로 JSON 포맷 + 명령어 품질을 동시 최적화

## 🚀 실행 방법

### 1. Vessl AI 워크스페이스 생성

1. [cloud.vessl.ai](https://cloud.vessl.ai) 접속
2. **H100 SXM x1** 선택 ($2.39/hr)
3. Container image: `Torch 2.9.1 (CUDA 13.0.1, Python 3.11)`
4. **Create** 클릭

### 2. SSH 접속 후 환경 설정

```bash
cd /root
git clone https://github.com/gyunggyung/Liquid-CLI.git
cd Liquid-CLI
bash setup_vessl.sh
```

### 3. HuggingFace 토큰 설정 (모델 업로드용)

```bash
# https://huggingface.co/settings/tokens 에서 Write 토큰 발급
export HF_TOKEN="hf_your_token_here"
huggingface-cli login --token $HF_TOKEN
```

### 4. 학습 실행 (~6-9시간, ~$15-22)

```bash
tmux new -s train

# Phase 0: 데이터 필터링 (~10분)
python prepare_data.py

# Phase 1: Full FT SFT (~3-5시간)
python train_sft.py --wandb --push_to_hub

# 중간 평가
python evaluate.py --model_path /root/outputs/sft/final

# Phase 2: GDPO RLVR (~2-4시간)
python train_gdpo.py --merge --wandb --push_to_hub

# Phase 3: GGUF 변환 + 업로드
python export_model.py --model_path /root/outputs/gdpo/merged --push_to_hub
```

### 5. 로컬 CPU에서 추론

```bash
./llama-cli -m LFM2-Terminal-Q8_0.gguf \
  -p "You are a terminal agent..." \
  --temp 0.3 -n 512 -t 8
```

### tmux 팁

```bash
Ctrl+B, D        # tmux에서 나가기 (학습은 계속됨)
tmux a -t train   # 다시 들어가기
```

## 📁 파일 구조

| 파일 | 역할 |
|------|------|
| `setup_vessl.sh` | 환경 설정 자동화 |
| `prepare_data.py` | 비코딩 5개 도메인 필터링 |
| `train_sft.py` | Phase 1: Full FT SFT (BF16 + DeepSpeed) |
| `train_gdpo.py` | Phase 2: GDPO RLVR (Unsloth FP8 + LoRA) |
| `evaluate.py` | 10개 태스크 자동 평가 |
| `export_model.py` | GGUF 변환 + Hub 업로드 |
| `ds_config.json` | DeepSpeed ZeRO-2 설정 |

자세한 내용은 [LFM2_Terminal_RLVR_Training_Guide.md](LFM2_Terminal_RLVR_Training_Guide.md)를 참고하세요.


## 📊 성능 평가 (Evaluation - Interim)

현재 **SFT Phase 1 (41-44% 진행)** 시점에서의 중간 평가 결과입니다.

| 지표 | 수치 | 비고 |
|------|------|------|
| **평균 점수** | 3.5 / 5.0 | 10개 태스크 종합 |
| **포맷 준수율** | 80% | JSON 구조 및 필수 필드 유지 |
| **명령어 관련성** | 70% | 태스크 해결에 필요한 명령어 포함 |

### 🔍 주요 관찰 사항
- **JSON 포맷 마스터**: 대부분의 태스크에서 4.5점 이상을 기록하며 터미널 에이전트로서의 출력 형식을 완벽하게 숙지함.
- **신중한 탐색(Exploration)**: 문제를 바로 해결하기 전 `ls`, `head`, `pwd` 등을 통해 환경을 먼저 확인하려는 실제 에이전트다운 지능적 패턴이 관찰됨.
- **아키텍처 안정성**: Liquid LFM2의 MoE 구조가 Unsloth 패치를 통해 H100 GPU에서 매우 안정적으로 학습 및 추론되고 있음을 확인.

---

# LFM2 Terminal Agent 🖥️ (English)

> Training LiquidAI's LFM2-8B-A1B as a terminal agent

## 🎯 Goal

Build a terminal agent that runs on a CPU laptop (16GB RAM).
Train on NVIDIA's [Nemotron-Terminal-Corpus](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus), convert to GGUF, run with llama.cpp.

## 💡 Why This Model?

- **LFM2-8B-A1B**: LNN (Liquid Neural Network) — gated short conv + GQA hybrid. 8B params but only 1B active → fast on CPU
- **Non-coding focus**: File ops, data processing, security, package management
- **GDPO**: Multi-reward RL optimizing JSON format + command quality simultaneously

## 🚀 Quick Start

```bash
# 1. Vessl AI: H100 SXM x1 ($2.39/hr)
#    Image: Torch 2.9.1 (CUDA 13.0.1, Python 3.11)

# 2. SSH in & setup
cd /root && git clone https://github.com/gyunggyung/Liquid-CLI.git
cd Liquid-CLI && bash setup_vessl.sh

# 3. HuggingFace token (for model upload)
export HF_TOKEN="hf_your_token_here"
huggingface-cli login --token $HF_TOKEN

# 4. Train (~6-9hrs, ~$15-22)
tmux new -s train
python prepare_data.py
python train_sft.py --wandb --push_to_hub
python evaluate.py --model_path /root/outputs/sft/final
python train_gdpo.py --merge --wandb --push_to_hub
python export_model.py --model_path /root/outputs/gdpo/merged --push_to_hub
```

## 📊 Evaluation Results (Interim)

Interim evaluation results at **SFT Phase 1 (41-44% Progress)**.

| Metric | Value | Notes |
|------|------|------|
| **Average Score** | 3.5 / 5.0 | Total across 10 tasks |
| **Format Compliance** | 80% | JSON structure & mandatory fields |
| **Command Relevance** | 70% | Inclusion of task-solving commands |

### 🔍 Key Observations
- **JSON Format Mastery**: Achieving 4.5+ scores in most tasks, showing perfect understanding of the terminal agent output schema.
- **Intelligent Exploration**: Observed patterns where the agent checks the environment (`ls`, `head`, `pwd`) before executing final commands, similar to human sysadmins.
- **Architectural Stability**: Successfully verified that the Liquid LFM2 MoE architecture is stable during training and inference on H100 via Unsloth patches.

## 📄 License

Apache License 2.0

## 🔗 References

- **Models & Data**
  - [Official LiquidAI LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B)
  - [Unsloth Optimized LFM2-8B-A1B](https://huggingface.co/unsloth/LFM2-8B-A1B)
  - [NVIDIA Nemotron-Terminal-Corpus Dataset](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus)
  - [Nemotron-Terminal Technical Report](https://arxiv.org/html/2602.21193v1)

- **Training Technologies**
  - [TRL v0.27.0 (GDPO/GRPO Implementation)](https://github.com/huggingface/trl/releases/tag/v0.27.0)
  - [GDPO (Generalized Differential Policy Optimization) Paper](https://arxiv.org/html/2601.05242v1)
  - [Unsloth Faster MoE kernels Guide](https://unsloth.ai/docs/basics/faster-moe)
  - [Unsloth FP8 RL & GRPO/GDPO Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning)

