import os

# DeepSpeed가 MPI/Slurm 환경으로 오해하지 않도록 모든 감지 변수 강제 삭제
keys_to_del = [k for k in os.environ.keys() if any(p in k for p in ["OMPI_", "PMI_", "MPI_", "MV2_", "SLURM_"])]
for k in keys_to_del:
    del os.environ[k]

import argparse
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

# train_sft.py와 동일한 시스템 프롬프트 유지
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

def main(args):
    print(f"🚀 토크나이징 완료본 생성 시작 (Model: {args.model_name})")
    
    # 1. 토크나이저 로딩
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 전처리된 텍스트 데이터 로드 또는 기존 토크나이징 완료본 로드
    if os.path.exists(args.output_path):
        print(f"✨ 이미 생성된 토크나이징 완료본을 발견했습니다: {args.output_path}")
        print("⚡ 토크나이징 과정을 건역뛰고 기존 데이터를 사용합니다.")
        tokenized_dataset = load_from_disk(args.output_path)
    else:
        print(f"📂 원본 데이터 로드 중: {args.input_path}")
        dataset = load_from_disk(args.input_path)
        
        def tokenize_function(example):
            # 텍스트 포맷팅 (train_sft.py 로직과 동일)
            messages = None
            for field in ["messages", "trajectory", "conversation", "turns"]:
                if field in example and example[field] is not None:
                    messages = example[field]
                    break
            
            if messages is not None and isinstance(messages, list):
                if len(messages) > 0 and isinstance(messages[0], dict) and "role" in messages[0]:
                    if messages[0].get("role") != "system":
                        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
                    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    full_text = str(example)
            elif "text" in example and example["text"]:
                full_text = example["text"]
            else:
                full_text = str(example)

            # 실제 토크나이징 (숫자로 변환)
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=args.max_seq_length,
                padding=False,
            )
            return tokenized

        print("⚡ 토크나이징 시작 (이 과정이 끝나면 앞으로는 대기 시간이 없습니다)...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            num_proc=16,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        print(f"💾 결과 저장 중: {args.output_path}")
        tokenized_dataset.save_to_disk(args.output_path)
    
    # 5. 허브 업로드 (선택 사항)
    if args.push_to_hub:
        print(f"📤 HuggingFace Hub 업로드 중: {args.hub_repo}")
        tokenized_dataset.push_to_hub(args.hub_repo, private=False)
        print(f"🎉 업로드 완료! 주소: https://huggingface.co/datasets/{args.hub_repo}")

    print("\n✅ 모든 작업이 완료되었습니다. 이제 이 데이터를 쓰면 1초 만에 학습이 시작됩니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/root/data/sft_data")
    parser.add_argument("--output_path", type=str, default="/root/data/sft_tokenized")
    parser.add_argument("--model_name", type=str, default="LiquidAI/LFM2-8B-A1B")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo", type=str, default="gyung/LFM2-Terminal-SFT-Tokenized")
    args = parser.parse_args()
    main(args)
