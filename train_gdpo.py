"""
train_gdpo.py — LFM2-8B-A1B GDPO RLVR 학습
TRL v0.27.0의 네이티브 GDPO 지원 사용
Vessl AI H100 SXM x1 환경

Usage:
    python train_gdpo.py --model_path /root/outputs/sft/final
    python train_gdpo.py --model_path /root/outputs/sft/final --max_steps 1000
"""

import argparse
import gc
import json
import os

import torch
from datasets import load_from_disk

# vLLM standby 모드 활성화 (메모리 공유)
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

# Unsloth MoE 최적화 백엔드 (H100: grouped_mm, A100: unsloth_triton)
# H100 SXM에서는 grouped_mm이 가장 빠름 (torch._grouped_mm 활용)
if "UNSLOTH_MOE_BACKEND" not in os.environ:
    os.environ["UNSLOTH_MOE_BACKEND"] = "grouped_mm"


# ═══════════════════════════════════════════════════
# 리워드 함수 (GDPO용 — 각각 독립 정규화됨)
# ═══════════════════════════════════════════════════


def _extract_text(completion):
    """TRL GRPOTrainer가 전달하는 completion에서 텍스트 추출.
    TRL 버전에 따라 str, list[dict], 또는 다른 형태일 수 있음.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) > 0:
        if isinstance(completion[0], dict):
            return completion[0].get("content", str(completion[0]))
        return str(completion[0])
    return str(completion)

# 유효한 터미널 명령어 목록
VALID_COMMANDS = [
    "ls", "cd", "cat", "grep", "find", "awk", "sed", "sort", "uniq",
    "chmod", "chown", "cp", "mv", "rm", "mkdir", "rmdir", "touch",
    "curl", "wget", "ssh", "scp", "ping", "netstat", "ifconfig", "ip",
    "pip", "apt", "npm", "conda", "brew", "yum", "dnf",
    "systemctl", "service", "ps", "kill", "top", "htop", "df", "du", "free",
    "tar", "zip", "unzip", "gzip", "gunzip",
    "head", "tail", "wc", "cut", "tr", "tee", "xargs",
    "jq", "diff", "patch", "echo", "export", "source", "which", "whereis",
    "whoami", "id", "su", "sudo", "passwd", "useradd", "usermod",
    "mount", "umount", "fdisk", "parted",
    "docker", "docker-compose", "kubectl",
    "git", "make", "cmake", "gcc", "g++",
    "python", "python3", "node", "npm", "npx",
    "crontab", "at", "nohup", "screen", "tmux",
    "openssl", "ssh-keygen", "gpg",
]


def reward_format(completions, **kwargs):
    """R1: JSON 포맷 준수 여부
    analysis, plan, commands 필드가 올바르게 포함되어 있는지 평가
    """
    scores = []
    for c in completions:
        resp = _extract_text(c)
        score = 0.0
        try:
            parsed = json.loads(resp)
            if isinstance(parsed, dict):
                if "analysis" in parsed and isinstance(parsed["analysis"], str):
                    score += 1.0
                if "plan" in parsed and isinstance(parsed["plan"], str):
                    score += 1.0
                if "commands" in parsed and isinstance(parsed["commands"], list):
                    score += 1.0
                    # 각 command가 올바른 구조인지
                    valid_cmds = sum(
                        1
                        for cmd in parsed["commands"]
                        if isinstance(cmd, dict) and "keystrokes" in cmd
                    )
                    if len(parsed["commands"]) > 0:
                        score += valid_cmds / len(parsed["commands"])
            else:
                score = -2.0
        except (json.JSONDecodeError, TypeError):
            score = -2.0
        scores.append(score)
    return scores


def reward_command_quality(completions, **kwargs):
    """R2: 명령어 품질
    실제 존재하는 Linux 명령어를 사용하는지, sudo 남용 없는지 등
    """
    scores = []
    for c in completions:
        resp = _extract_text(c)
        score = 0.0
        try:
            parsed = json.loads(resp)
            cmds = parsed.get("commands", [])
            if not cmds:
                scores.append(-0.5)
                continue

            for cmd_obj in cmds:
                ks = cmd_obj.get("keystrokes", "").strip()
                if not ks:
                    score -= 0.3
                    continue

                # 첫 번째 단어 (명령어) 추출
                first_word = ks.split()[0].rstrip("\n") if ks.split() else ""
                # sudo 뒤의 실제 명령어도 체크
                if first_word == "sudo" and len(ks.split()) > 1:
                    first_word = ks.split()[1].rstrip("\n")

                if first_word in VALID_COMMANDS:
                    score += 0.5
                elif first_word.startswith("./") or first_word.startswith("/"):
                    score += 0.3  # 절대/상대 경로 실행
                elif first_word:
                    score -= 0.3  # 알 수 없는 명령어

                # duration이 합리적인지
                duration = cmd_obj.get("duration", 1.0)
                if isinstance(duration, (int, float)) and 0.05 <= duration <= 60:
                    score += 0.1
                else:
                    score -= 0.1

                # newline으로 끝나는지 (실행을 위해)
                if ks.endswith("\n") or ks.endswith("\\n"):
                    score += 0.1

        except (json.JSONDecodeError, TypeError, AttributeError):
            score = -1.0
        scores.append(score)
    return scores


def reward_reasoning(completions, **kwargs):
    """R3: 추론 품질
    analysis와 plan이 의미 있는 내용을 담고 있는지
    """
    TERMINAL_KEYWORDS = [
        "file", "directory", "folder", "command", "output", "error",
        "install", "run", "execute", "check", "verify", "list",
        "create", "delete", "modify", "update", "search", "find",
        "permission", "process", "service", "package", "config",
        "path", "script", "log", "port", "network", "disk",
    ]

    scores = []
    for c in completions:
        resp = _extract_text(c)
        score = 0.0
        try:
            parsed = json.loads(resp)
            analysis = parsed.get("analysis", "")
            plan = parsed.get("plan", "")

            # analysis 품질
            analysis_words = len(analysis.split()) if analysis else 0
            if 15 <= analysis_words <= 200:
                score += 1.0
            elif 5 <= analysis_words < 15:
                score += 0.3
            elif analysis_words > 200:
                score -= 0.3  # 과도하게 긴 분석
            else:
                score -= 0.5

            # plan 품질
            plan_words = len(plan.split()) if plan else 0
            if 10 <= plan_words <= 150:
                score += 1.0
            elif plan_words < 10:
                score -= 0.3
            else:
                score -= 0.2

            # 터미널 관련 키워드 포함 여부
            combined = (analysis + " " + plan).lower()
            keyword_hits = sum(1 for kw in TERMINAL_KEYWORDS if kw in combined)
            score += min(keyword_hits * 0.15, 1.0)

        except (json.JSONDecodeError, TypeError):
            score = -1.0
        scores.append(score)
    return scores


def reward_length(completions, **kwargs):
    """R4: 응답 길이 제어
    너무 짧거나 너무 긴 응답에 페널티
    """
    scores = []
    for c in completions:
        resp = _extract_text(c)
        char_len = len(resp)
        # 이상적 범위: 200~2000 characters
        if 200 <= char_len <= 2000:
            scores.append(1.0)
        elif 100 <= char_len < 200 or 2000 < char_len <= 3000:
            scores.append(0.5)
        elif char_len > 3000:
            scores.append(-0.5 * min(char_len / 3000, 3.0))
        elif char_len < 50:
            scores.append(-1.0)
        else:
            scores.append(0.0)
    return scores


# ═══════════════════════════════════════════════════
# RLVR 데이터 변환
# ═══════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{
  "analysis": "Analyze the current state based on the terminal output provided.",
  "plan": "Describe your plan for the next steps.",
  "commands": [
    {"keystrokes": "ls -la\\n", "duration": 0.1}
  ],
  "task_complete": false
}"""


def convert_to_rlvr_format(dataset, tokenizer):
    """SFT 데이터를 RLVR 프롬프트 형식으로 변환"""

    def extract_prompt(example):
        """각 trajectory에서 첫 번째 (system + user) 프롬프트 추출"""
        messages = None
        for field in ["messages", "trajectory", "conversation", "turns"]:
            if field in example and example[field] is not None:
                messages = example[field]
                break

        prompt_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        answer = ""

        if messages and isinstance(messages, list):
            for i, msg in enumerate(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    prompt_messages.append(
                        {"role": "user", "content": msg["content"]}
                    )
                    # 다음 assistant 응답이 정답
                    if (
                        i + 1 < len(messages)
                        and isinstance(messages[i + 1], dict)
                        and messages[i + 1].get("role") == "assistant"
                    ):
                        answer = messages[i + 1]["content"]
                    break
        elif "input" in example:
            prompt_messages.append({"role": "user", "content": example["input"]})
            answer = example.get("output", "")

        return {"prompt": prompt_messages, "answer": answer}

    return dataset.map(extract_prompt, num_proc=4, desc="Converting to RLVR format")


def main(args):
    print("=" * 60)
    print("Phase 2: GDPO RLVR — LFM2-8B-A1B Terminal Agent")
    print("=" * 60)
    print(f"  Base model: {args.model_path}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Max steps: {args.max_steps}")

    # ─── 모델 로딩 (SFT 결과 + LoRA) ───
    print("\n[1/4] SFT 모델 로딩 + LoRA 적용...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_fp8=True,
        fast_inference=True,  # vLLM 활성화 (RLVR rollout 생성)
        max_lora_rank=args.lora_rank,
        trust_remote_code=True,
    )

    # RLVR에서는 LoRA 적용
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules="all-linear",
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─── 데이터 로딩 + RLVR 포맷 변환 ───
    print("\n[2/4] 데이터 로딩 + RLVR 포맷 변환...")
    dataset = load_from_disk(args.data_path)
    rlvr_data = convert_to_rlvr_format(dataset, tokenizer)

    # 프롬프트 길이 필터링
    rlvr_data = rlvr_data.filter(
        lambda x: x["prompt"] and len(x["prompt"]) >= 2,
        desc="Filtering valid prompts",
    )
    print(f"  RLVR 데이터: {len(rlvr_data)} samples")

    # ─── GDPO 학습 설정 ───
    print("\n[3/4] GDPO 학습 시작...")
    from trl import GRPOConfig, GRPOTrainer
    from vllm import SamplingParams

    vllm_params = SamplingParams(
        min_p=0.1,
        top_p=0.95,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    training_args = GRPOConfig(
        # vLLM 추론 활성화
        use_vllm=True,
        vllm_sampling_params=vllm_params,
        temperature=1.0,
        # 학습률
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        # 배치 (H100 1대)
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        # 시퀀스 길이
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        # 학습 스텝
        max_steps=args.max_steps,
        save_steps=100,
        logging_steps=1,
        # KL 계수 (SFT 모델을 reference로)
        kl_coef=args.kl_coef,
        # GDPO: 각 reward를 독립적으로 group-level 정규화 후 합산
        # TRL v0.27.0+ loss_type 파라미터로 활성화
        loss_type="gdpo",
        # HuggingFace Hub (중간 체크포인트 자동 업로드)
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo if args.push_to_hub else None,
        hub_strategy="every_save",  # save_steps마다 자동 업로드
        # 출력
        report_to="wandb" if args.wandb else "none",
        output_dir=args.output_dir,
        run_name="lfm2-terminal-gdpo",
        seed=3407,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_format,          # R1: JSON 포맷 준수
            reward_command_quality,  # R2: 명령어 품질
            reward_reasoning,       # R3: 추론 품질
            reward_length,          # R4: 응답 길이 제어
        ],
        args=training_args,
        train_dataset=rlvr_data,
    )

    trainer.train()

    # ─── 저장 ───
    print("\n[4/4] 모델 저장...")
    lora_path = os.path.join(args.output_dir, "lora_adapter")
    model.save_lora(lora_path)
    print(f"  LoRA 어댑터 저장: {lora_path}")

    # LoRA 검증
    from safetensors import safe_open

    adapter_file = os.path.join(lora_path, "adapter_model.safetensors")
    if os.path.exists(adapter_file):
        with safe_open(adapter_file, framework="pt") as f:
            for key in list(f.keys())[:5]:
                tensor = f.get_tensor(key)
                n_zeros = (tensor == 0).sum().item() / tensor.numel()
                if n_zeros == 1.0:
                    print(f"  ⚠️ WARNING: Layer {key} is all zeros!")
        print("  ✅ LoRA 어댑터 검증 완료")

    # 머지된 모델 저장 (선택)
    if args.merge:
        merged_path = os.path.join(args.output_dir, "merged")
        model.save_pretrained_merged(
            merged_path, tokenizer, save_method="merged_16bit"
        )
        print(f"  ✅ 머지된 모델 저장: {merged_path}")

    print(f"\n다음 단계:")
    print(f"  평가:     python evaluate.py --model_path {lora_path}")
    print(f"  변환:     python export_model.py --model_path {merged_path if args.merge else lora_path}")

    # HuggingFace Hub 최종 업로드 (중간 체크포인트는 hub_strategy="every_save"로 자동 업로드됨)
    if args.push_to_hub:
        upload_path = merged_path if args.merge else lora_path
        print(f"\n  📤 HuggingFace Hub 최종 모델 업로드: {args.hub_repo}")
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=upload_path,
            repo_id=args.hub_repo,
            repo_type="model",
            commit_message="GDPO RLVR: Terminal agent with 4 reward functions — final",
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"  ✅ 업로드 완료: https://huggingface.co/{args.hub_repo}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LFM2-8B-A1B GDPO RLVR")
    parser.add_argument("--model_path", type=str, default="/root/outputs/sft/final")
    parser.add_argument("--data_path", type=str, default="/root/data/sft_data")
    parser.add_argument("--output_dir", type=str, default="/root/outputs/gdpo")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--merge", action="store_true", default=False)
    parser.add_argument("--push_to_hub", action="store_true", default=False,
                        help="학습 후 HuggingFace Hub에 업로드")
    parser.add_argument("--hub_repo", type=str, default="gyung/LFM2-8B-Terminal-GDPO",
                        help="HuggingFace 리포지토리 ID")
    args = parser.parse_args()
    main(args)
