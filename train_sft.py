"""
train_sft.py — LFM2-8B-A1B Full Fine-Tuning SFT
Vessl AI H100 SXM x1 환경

Usage:
    python train_sft.py --data_path /root/data/sft_data
"""
import os

# DeepSpeed가 MPI/Slurm 환경으로 오해하지 않도록 모든 감지 변수 강제 삭제
keys_to_del = [k for k in os.environ.keys() if any(p in k for p in ["OMPI_", "PMI_", "MPI_", "MV2_", "SLURM_"])]
for k in keys_to_del:
    del os.environ[k]

# 단일 GPU 학습을 위해 분산 학습용 변수들 명시적 고정
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

# DeepSpeed/Cuda 백엔드 강제 고정
os.environ["DS_SKIP_CUDA_CHECK"] = "1"
os.environ["DS_ACCELERATOR"] = "cuda"
os.environ["DS_COMM_BACKEND"] = "nccl"

import argparse
import gc
import json
import torch
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {"keystrokes": "ls -la\\n", "duration": 0.1},
    {"keystrokes": "cd project\\n", "duration": 0.1}
  ],
  "task_complete": false
}"""

def main(args):
    print("=" * 60)
    print("Phase 1: Full FT SFT — LFM2-8B-A1B Terminal Agent")
    print("=" * 60)

    # 1. 모델 및 토크나이저 로드
    print("\n[1/4] 모델 로딩 (Full FT → BF16)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None, # DeepSpeed 제어
        attn_implementation="sdpa",
    )

    # 2. 데이터 로딩 (전처리 완료본 우선 확인)
    print("\n[2/4] 데이터 로딩...")
    tokenized_path = "/root/data/sft_tokenized"
    
    if os.path.exists(tokenized_path):
        print(f"🚀 토크나이징 완료본 발견! 대기 시간 없이 즉시 실행합니다: {tokenized_path}")
        dataset = load_from_disk(tokenized_path)
        dataset_text_field = None # 이미 토크나이징됨
    else:
        print(f"📂 원본 데이터 로드 및 포맷팅 중 (대기 시간 약 8분 예상): {args.data_path}")
        dataset = load_from_disk(args.data_path)
        # 텍스트 필드가 없는 경우를 위해 포맷팅 로직 (필요시 호출)
        if "text" not in dataset.column_names:
            print("  포맷 변환 중...")
            # 여기서는 간단히 문자열 변환만 예시로 둠 (실제 필요시 보완)
        dataset_text_field = "text"

    print(f"  Dataset Loaded: {len(dataset)} samples")

    # 3. 학습 설정
    print("\n[3/4] 학습 시작...")
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_length,
        dataset_text_field=dataset_text_field,
        dataset_num_proc=16,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        bf16=True,
        packing=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=500,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo if args.push_to_hub else None,
        deepspeed=args.deepspeed_config,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # 학습 재개 확인
    resume_from = None
    if args.resume and os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            resume_from = os.path.join(args.output_dir, latest)
            print(f"  Resuming from: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    # 4. 저장
    print("\n[4/4] 모델 저장...")
    save_path = os.path.join(args.output_dir, "final")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ 완료: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/root/data/sft_data")
    parser.add_argument("--model_name", type=str, default="LiquidAI/LFM2-8B-A1B")
    parser.add_argument("--output_dir", type=str, default="/root/outputs/sft")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    parser.add_argument("--push_to_hub", action="store_true", default=True)
    parser.add_argument("--hub_repo", type=str, default="gyung/LFM2-8B-Terminal-SFT")
    args = parser.parse_args()
    main(args)
