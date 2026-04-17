import builtins
import transformers

# Unsloth 내부 exec() 환경에서도 작동하도록 builtins에 강제 주입
if not hasattr(builtins, "PreTrainedConfig"):
    if hasattr(transformers, "PreTrainedConfig"):
        builtins.PreTrainedConfig = transformers.PreTrainedConfig
    elif hasattr(transformers, "PretrainedConfig"):
        builtins.PreTrainedConfig = transformers.PretrainedConfig

from unsloth import FastLanguageModel
import torch
import os
import argparse
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import standardize_data_formats, train_on_responses_only

def main(args):
    print("============================================================")
    print("Phase 1: Unsloth Full FT SFT — LFM2-8B-A1B Terminal Agent")
    print("============================================================")

    # 1. 모델 로딩 (Unsloth 최적화 커널 로드)
    # full_finetuning=True를 사용하여 LoRA 없이 전체 가중치 학습
    print(f"\n[1/4] Unsloth 모델 로딩 (Full FT 모드)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        load_in_4bit = False, # Full FT이므로 양자화 안 함
        full_finetuning = True,
        trust_remote_code = True,
    )

    # 2. 데이터 로딩 및 포맷팅
    print(f"\n[2/4] 데이터 로딩 및 포맷팅: {args.data_path}")
    dataset = load_from_disk(args.data_path)
    
    # Unsloth 표준 데이터 포맷으로 변환 (Nemotron ShareGPT 대응)
    dataset = standardize_data_formats(dataset)

    def formatting_prompts_func(examples):
        texts = tokenizer.apply_chat_template(
            examples["conversations"],
            tokenize = False,
            add_generation_prompt = False,
        )
        # BOS 토큰 중복 방지
        return { "text" : [x.removeprefix(tokenizer.bos_token) for x in texts] }

    dataset = dataset.map(formatting_prompts_func, batched = True, num_proc = 16)

    # 3. 학습 설정 (Unsloth 최적화 엔진 결합)
    print(f"\n[3/4] 학습 시작 (Unsloth Turbo Engine)...")
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = args.max_seq_length,
        packing = True, # 패킹 활성화로 속도 2배 향상
        args = SFTConfig(
            output_dir = args.output_dir,
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.grad_accum,
            warmup_steps = 10,
            learning_rate = args.learning_rate,
            num_train_epochs = args.num_epochs,
            max_steps = args.max_steps,
            save_steps = 50,
            save_total_limit = 2,
            hub_strategy = "checkpoint",
            optim = "adamw_8bit", # 아담 8비트로 메모리 추가 확보
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            bf16 = True, # H100/A100 필수
            report_to = "none",
            push_to_hub = args.push_to_hub,
            hub_model_id = args.hub_repo,
        ),
    )

    # Assistant 응답 부분에 대해서만 Loss 계산 (정확도 향상)
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )

    # 학습 시작
    trainer.train()

    # 4. 모델 저장
    print(f"\n[4/4] 모델 저장 중: {args.output_dir}")
    # Full FT 모델은 일반 save_pretrained로 저장 가능
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if args.push_to_hub:
        print(f"🚀 허깅페이스 업로드 중: {args.hub_repo}")
        model.push_to_hub(args.hub_repo)
        tokenizer.push_to_hub(args.hub_repo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/root/data/sft_data")
    parser.add_argument("--model_name", type=str, default="LiquidAI/LFM2-8B-A1B")
    parser.add_argument("--output_dir", type=str, default="/root/outputs/sft_unsloth")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--hub_repo", type=str, default="gyung/LFM2-8B-Terminal-SFT-Unsloth")
    
    args = parser.parse_args()
    main(args)
