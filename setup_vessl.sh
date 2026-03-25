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
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo
pip install -r requirements.txt

echo "═══ [4/5] wandb 로그인 (선택) ═══"
echo "wandb login 필요시: wandb login YOUR_API_KEY"

echo "═══ [5/5] GPU 정보 확인 ═══"
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
python -c "import unsloth; print(f'Unsloth OK')"
python -c "import trl; print(f'TRL: {trl.__version__}')"

echo ""
echo "═══ 설정 완료! ═══"
echo "다음 단계:"
echo "  1. 데이터 준비: python prepare_data.py"
echo "  2. SFT 학습:    python train_sft.py"
echo "  3. 평가:        python evaluate.py --model_path outputs/sft"
echo "  4. GDPO RLVR:   python train_gdpo.py"
echo "  5. 모델 변환:   python export_model.py"
