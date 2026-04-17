"""
train_sft.py — LFM2-8B-A1B Full Fine-Tuning SFT
Vessl AI H100 SXM x1 환경

Usage:
    python train_sft.py --data_path /root/data/sft_data
    python train_sft.py --data_path /root/data/sft_data --max_steps 500 --resume
"""

import argparse
import gc
import json
import os

import torch
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer

# ═══════════════════════════════════════════════════
# Terminus 2 Agent System Prompt (터미널 에이전트 포맷)
# ═══════════════════════════════════════════════════
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
}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete (defaults to 1.0)

IMPORTANT: Most bash commands should end with a newline (\\n) to cause them to execute."""


def format_messages_to_text(example, tokenizer):
    """데이터셋의 trajectory를 SFT 학습용 텍스트로 변환

    Nemotron-Terminal-Corpus의 실제 필드 구조에 따라 자동 적응.
    """
    # 가능한 필드명들
    messages = None
    for field in ["messages", "trajectory", "conversation", "turns"]:
        if field in example and example[field] is not None:
            messages = example[field]
            break

    if messages is not None and isinstance(messages, list):
        # messages가 이미 [{"role": ..., "content": ...}] 형태인 경우
        if (
            len(messages) > 0
            and isinstance(messages[0], dict)
            and "role" in messages[0]
        ):
            # system prompt 추가 (없으면)
            if messages[0].get("role") != "system":
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ] + messages
            return tokenizer.apply_chat_template(messages, tokenize=False)

    # "text" 필드가 이미 포맷된 텍스트인 경우
    if "text" in example and example["text"]:
        return example["text"]

    # "input"/"output" 구조인 경우
    if "input" in example and "output" in example:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # fallback: 모든 필드를 문자열로 연결
    return str(example)


def main(args):
    print("=" * 60)
    print("Phase 1: Full FT SFT — LFM2-8B-A1B Terminal Agent")
    print("=" * 60)
    print(f"  Data: {args.data_path}")
    print(f"  Model: {args.model_name}")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  DeepSpeed: {args.deepspeed_config}")

    # ─── 모델 로딩 ───
    # ⚠️ 중요: Unsloth FP8는 LoRA 학습에 최적화되어 있음
    # Full FT는 transformers + BF16으로 진행 (DeepSpeed로 OOM 방지)
    # LFM2는 LNN(Liquid Neural Network) 아키텍처: gated short conv + GQA 하이브리드
    print("\n[1/4] 모델 로딩 (Full FT → BF16 + DeepSpeed)...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype="bfloat16",  # LFM2 공식 model card 방식
        trust_remote_code=True,
        # device_map은 DeepSpeed 사용 시 생략해야 함
        device_map="auto" if not args.deepspeed_config else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded: {type(model).__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} (Full FT)")

    # ─── 데이터 로딩 ───
    print("\n[2/4] 데이터 로딩...")
    dataset = load_from_disk(args.data_path)
    print(f"  Dataset: {len(dataset)} samples")
    print(f"  Columns: {dataset.column_names}")

    # 텍스트 포맷팅
    print("  포맷 변환 중...")
    dataset = dataset.map(
        lambda x: {"text": format_messages_to_text(x, tokenizer)},
        num_proc=4,
        desc="Formatting",
    )

    # 길이 필터링
    dataset = dataset.map(
        lambda x: {"_len": len(tokenizer.encode(x["text"]))},
        num_proc=4,
        desc="Tokenizing lengths",
    )
    max_len = int(
        min(
            args.max_seq_length,
            max(dataset["_len"]) if max(dataset["_len"]) < args.max_seq_length else args.max_seq_length,
        )
    )
    original_len = len(dataset)
    dataset = dataset.filter(lambda x: x["_len"] <= args.max_seq_length)
    print(
        f"  길이 필터링: {original_len} → {len(dataset)} "
        f"(제거: {original_len - len(dataset)})"
    )

    # 샘플 확인
    print(f"\n  첫 번째 샘플 (처음 300자):")
    print(f"  {dataset[0]['text'][:300]}...")

    # ─── 학습 설정 ───
    print("\n[3/4] 학습 시작...")

    training_args = SFTConfig(
        # 데이터
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=True,  # 짧은 시퀀스를 묶어서 효율 향상
        # 배치
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # 최적화
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        # 스케줄
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        # 메모리
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        # Deepspeed (OOM 방지)
        deepspeed=args.deepspeed_config if args.deepspeed_config else None,
        # HuggingFace Hub (중간 체크포인트 자동 업로드)
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo if args.push_to_hub else None,
        hub_strategy="every_save",  # save_steps마다 자동 업로드
        # 출력
        report_to="wandb" if args.wandb else "none",
        output_dir=args.output_dir,
        run_name="lfm2-terminal-sft",
        seed=3407,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # 학습 재개
    resume_from = None
    if args.resume and os.path.exists(args.output_dir):
        checkpoints = [
            d
            for d in os.listdir(args.output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            resume_from = os.path.join(args.output_dir, latest)
            print(f"  Resuming from: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    # ─── 저장 ───
    print("\n[4/4] 모델 저장...")
    save_path = os.path.join(args.output_dir, "final")

    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"  ✅ 모델 저장 완료: {save_path}")

    # HuggingFace Hub 최종 업로드 (중간 체크포인트는 hub_strategy="every_save"로 자동 업로드됨)
    if args.push_to_hub:
        print(f"\n  📤 HuggingFace Hub 최종 모델 업로드: {args.hub_repo}")
        trainer.push_to_hub(commit_message="SFT: Full FT on Nemotron-Terminal-Corpus (non-coding) — final")
        print(f"  ✅ 업로드 완료: https://huggingface.co/{args.hub_repo}")

    print(f"\n다음 단계:")
    print(f"  평가:     python evaluate.py --model_path {save_path}")
    print(f"  GDPO RL:  python train_gdpo.py --model_path {save_path}")

    # 메모리 정리
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LFM2-8B-A1B Full FT SFT")
    parser.add_argument("--data_path", type=str, default="/root/data/sft_data")
    parser.add_argument(
        "--model_name", type=str, default="unsloth/LFM2-8B-A1B"
    )
    parser.add_argument("--output_dir", type=str, default="/root/outputs/sft")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    parser.add_argument("--push_to_hub", action="store_true", default=False,
                        help="학습 후 HuggingFace Hub에 업로드")
    parser.add_argument("--hub_repo", type=str, default="gyung/LFM2-8B-Terminal-SFT",
                        help="HuggingFace 리포지토리 ID")
    args = parser.parse_args()
    main(args)
