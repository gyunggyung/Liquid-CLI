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

### Vessl AI H100에서 (권장)

```bash
# 1. 워크스페이스 생성
#    → H100 SXM x1 선택 ($2.39/hr)
#    → Managed image: Torch 2.9.1 (CUDA 13.0.1, Python 3.11)
#    → Persistent volume 추가!

# 2. SSH 접속 후
cd /root
git clone https://github.com/gyunggyung/Liquid-CLI.git
cd Liquid-CLI
bash setup_vessl.sh

# 3. 학습 (~6-9시간, ~$15-22)
tmux new -s train
python prepare_data.py                    # 데이터 필터링 (~10분)
python train_sft.py --wandb               # Full FT SFT (~3-5시간)
python evaluate.py --model_path /root/outputs/sft/final
python train_gdpo.py --merge --wandb      # GDPO RLVR (~2-4시간)
python export_model.py --model_path /root/outputs/gdpo/merged  # GGUF 변환
```

### 로컬 CPU에서 추론

```bash
./llama-cli -m LFM2-Terminal-Q8_0.gguf \
  -p "You are a terminal agent..." \
  --temp 0.3 -n 512 -t 8
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

### On Vessl AI H100 (Recommended)

```bash
# 1. Create workspace: H100 SXM x1 ($2.39/hr)
#    Image: Torch 2.9.1 (CUDA 13.0.1, Python 3.11)
#    Add persistent volume!

# 2. SSH in
cd /root && git clone https://github.com/gyunggyung/Liquid-CLI.git
cd Liquid-CLI && bash setup_vessl.sh

# 3. Train (~6-9hrs, ~$15-22)
tmux new -s train
python prepare_data.py
python train_sft.py --wandb
python evaluate.py --model_path /root/outputs/sft/final
python train_gdpo.py --merge --wandb
python export_model.py --model_path /root/outputs/gdpo/merged
```

## 📄 License

MIT

## 🔗 References

- [Nemotron-Terminal Paper](https://arxiv.org/html/2602.21193v1)
- [LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B)
- [TRL v0.27.0 (GDPO)](https://github.com/huggingface/trl/releases/tag/v0.27.0)
- [GDPO Paper](https://arxiv.org/html/2601.05242v1)
