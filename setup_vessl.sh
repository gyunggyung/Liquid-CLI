#!/bin/bash
# ═══════════════════════════════════════════════════
# Vessl AI H100 초기 설정 스크립트
# Managed Image: Torch 2.9.1 (CUDA 13.0.1, Python 3.11)
# ═══════════════════════════════════════════════════

set -e

echo "═══ [1/5] 시스템 패키지 업데이트 ═══"
apt-get update && apt-get install -y git curl wget tmux htop

echo "═══ [2/5] 프로젝트 클론 ═══"
cd /root
if [ ! -d "Liquid-CLI" ]; then
    git clone https://github.com/gyunggyung/Liquid-CLI.git
fi
cd Liquid-CLI

echo "═══ [3/5] Python 패키지 설치 ═══"
pip install --upgrade pip
# LFM2 모델 인식을 위한 특정 transformers 버전(lfm2_moe 지원) 설치 필수
pip install git+https://github.com/huggingface/transformers.git@0c9a72e4576fe4c84077f066e585129c97bfd4e6
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo
pip install -r requirements.txt

echo "═══ [4/6] HuggingFace 토큰 설정 ═══"
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token $HF_TOKEN
    echo "✅ HF 토큰 설정 완료"
else
    echo "⚠️  HF_TOKEN이 설정되지 않았습니다."
    echo "  모델 업로드를 원하면: export HF_TOKEN=hf_your_token_here"
    echo "  토큰 발급: https://huggingface.co/settings/tokens"
fi

echo "═══ [5/6] wandb 로그인 (선택) ═══"
echo "wandb login 필요시: wandb login YOUR_API_KEY"

echo "═══ [6/6] GPU 정보 확인 ═══"
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
python -c "import unsloth; print(f'Unsloth OK')"
python -c "import trl; print(f'TRL: {trl.__version__}')"

echo ""
echo "═══ 설정 완료! ═══"
echo ""
echo "HF 토큰 설정 (모델 업로드용):"
echo "  export HF_TOKEN=hf_your_token_here"
echo ""
echo "다음 단계:"
echo "  1. 데이터 준비: python prepare_data.py"
echo "  2. SFT 학습:    python train_sft.py --wandb --push_to_hub"
echo "  3. 평가:        python evaluate.py --model_path /root/outputs/sft/final"
echo "  4. GDPO RLVR:   python train_gdpo.py --merge --wandb --push_to_hub"
echo "  5. 모델 변환:   python export_model.py --model_path /root/outputs/gdpo/merged --push_to_hub"
