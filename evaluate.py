"""
evaluate.py — 학습된 LFM2 터미널 에이전트 평가

Usage:
    python evaluate.py --model_path /root/outputs/sft/final
    python evaluate.py --model_path /root/outputs/gdpo/lora_adapter
"""

import argparse
import json
import os

import torch


# ═══════════════════════════════════════════════════
# 평가 프롬프트 (비코딩 터미널 태스크)
# ═══════════════════════════════════════════════════
EVAL_PROMPTS = [
    {
        "id": "file_ops_1",
        "category": "file_operations",
        "prompt": "List all files larger than 100MB in the /home directory, sorted by size in descending order.",
        "expected_commands": ["find", "sort"],
    },
    {
        "id": "file_ops_2",
        "category": "file_operations",
        "prompt": "Find all .log files modified in the last 7 days under /var/log and count the total number of lines across all of them.",
        "expected_commands": ["find", "wc", "xargs"],
    },
    {
        "id": "data_proc_1",
        "category": "data_processing",
        "prompt": "Extract the 3rd column from a CSV file /data/input.csv, sort the values, remove duplicates, and save to /data/output.txt",
        "expected_commands": ["cut", "sort", "uniq"],
    },
    {
        "id": "data_proc_2",
        "category": "data_processing",
        "prompt": "Replace all occurrences of 'ERROR' with 'WARNING' in all .txt files under /logs/ directory recursively.",
        "expected_commands": ["find", "sed"],
    },
    {
        "id": "data_query_1",
        "category": "data_querying",
        "prompt": "From the JSON file /data/users.json, extract all users with age greater than 30 and display only their names.",
        "expected_commands": ["jq", "cat"],
    },
    {
        "id": "dep_mgmt_1",
        "category": "dependency_management",
        "prompt": "Check which Python packages are outdated and list them. Then upgrade pip itself to the latest version.",
        "expected_commands": ["pip", "list"],
    },
    {
        "id": "dep_mgmt_2",
        "category": "dependency_management",
        "prompt": "Install nginx via apt, enable it as a systemd service, and verify it's running.",
        "expected_commands": ["apt", "systemctl"],
    },
    {
        "id": "security_1",
        "category": "security",
        "prompt": "Check the current firewall rules, then add a rule to allow incoming traffic on port 443 (HTTPS).",
        "expected_commands": ["iptables", "ufw"],
    },
    {
        "id": "network_1",
        "category": "networking",
        "prompt": "Download a file from https://example.com/data.tar.gz, verify its checksum, and extract it to /opt/data/",
        "expected_commands": ["wget", "curl", "tar", "sha256sum", "md5sum"],
    },
    {
        "id": "sysadmin_1",
        "category": "system_admin",
        "prompt": "Check disk usage of all mounted filesystems, identify which partition has the most usage, and find the top 10 largest files on that partition.",
        "expected_commands": ["df", "du", "sort", "find"],
    },
]

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


def evaluate_response(response: str, expected_commands: list) -> dict:
    """단일 응답 평가"""
    result = {
        "format_ok": False,
        "json_valid": False,
        "has_analysis": False,
        "has_plan": False,
        "has_commands": False,
        "commands_relevant": False,
        "command_names": [],
        "score": 0.0,
    }

    # JSON 파싱
    try:
        parsed = json.loads(response)
        result["json_valid"] = True
        result["score"] += 1.0
    except (json.JSONDecodeError, TypeError):
        # JSON이 아닌 경우 부분 파싱 시도
        try:
            json_match = response[response.index("{") : response.rindex("}") + 1]
            parsed = json.loads(json_match)
            result["json_valid"] = True
            result["score"] += 0.5
        except (ValueError, json.JSONDecodeError):
            return result

    # 필드 확인
    if "analysis" in parsed and parsed["analysis"]:
        result["has_analysis"] = True
        result["score"] += 1.0
    if "plan" in parsed and parsed["plan"]:
        result["has_plan"] = True
        result["score"] += 1.0
    if "commands" in parsed and isinstance(parsed["commands"], list) and parsed["commands"]:
        result["has_commands"] = True
        result["score"] += 1.0

        # 명령어 추출
        for cmd in parsed["commands"]:
            ks = cmd.get("keystrokes", "").strip()
            if ks:
                first_word = ks.split()[0].rstrip("\n")
                if first_word == "sudo" and len(ks.split()) > 1:
                    first_word = ks.split()[1].rstrip("\n")
                result["command_names"].append(first_word)

        # 기대하는 명령어가 포함되어 있는지
        for expected in expected_commands:
            if expected in result["command_names"]:
                result["commands_relevant"] = True
                result["score"] += 1.0
                break

    result["format_ok"] = all(
        [result["json_valid"], result["has_analysis"], result["has_plan"], result["has_commands"]]
    )
    return result


def main(args):
    print("=" * 60)
    print("LFM2 Terminal Agent 평가")
    print("=" * 60)

    # SFT 단계에서 transformers로 저장한 모델은 transformers로 로딩
    # GDPO 단계에서 Unsloth로 저장한 LoRA는 Unsloth로 로딩
    # LFM2는 LNN(Liquid Neural Network): gated short conv + GQA 하이브리드
    is_unsloth = args.use_unsloth

    # 자동 감지: adapter_config.json이 있으면 LoRA → Unsloth 사용
    adapter_path = os.path.join(args.model_path, "adapter_config.json")
    if os.path.exists(adapter_path) and not is_unsloth:
        print("  adapter_config.json 감지 → Unsloth로 전환")
        is_unsloth = True

    if is_unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=args.max_seq_length,
            load_in_fp8=True,
            trust_remote_code=True,
        )
        FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype="bfloat16", device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model.eval()

    # ─── 평가 ───
    results = []
    category_scores = {}

    for test in EVAL_PROMPTS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Current terminal state:\n$ \n\nTask: {test['prompt']}",
            },
        ]

        # LFM2 공식 model card 방식: apply_chat_template → tensor 직접 반환
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            output = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.3,      # LFM2 공식 권장값
                min_p=0.15,           # LFM2 공식 권장값
                repetition_penalty=1.05,  # LFM2 공식 권장값
                max_new_tokens=1024,
            )

        response = tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        )

        eval_result = evaluate_response(response, test["expected_commands"])
        eval_result["id"] = test["id"]
        eval_result["category"] = test["category"]
        eval_result["response_preview"] = response[:200]
        results.append(eval_result)

        # 카테고리별 집계
        cat = test["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(eval_result["score"])

        # 진행 상황 출력
        status = "✅" if eval_result["format_ok"] else "❌"
        print(
            f"  {status} [{test['id']}] score={eval_result['score']:.1f} "
            f"cmds={eval_result['command_names']}"
        )

    # ─── 결과 요약 ───
    print("\n" + "=" * 60)
    print("평가 결과 요약")
    print("=" * 60)

    total_scores = [r["score"] for r in results]
    format_ok_count = sum(1 for r in results if r["format_ok"])
    relevant_count = sum(1 for r in results if r["commands_relevant"])

    print(f"\n전체 결과:")
    print(f"  총 평가: {len(results)}")
    print(f"  평균 점수: {sum(total_scores)/len(total_scores):.2f} / 5.0")
    print(f"  포맷 준수율: {format_ok_count}/{len(results)} ({format_ok_count/len(results):.1%})")
    print(f"  관련 명령어 사용율: {relevant_count}/{len(results)} ({relevant_count/len(results):.1%})")

    print(f"\n카테고리별 결과:")
    for cat, scores in sorted(category_scores.items()):
        avg = sum(scores) / len(scores)
        print(f"  {cat}: {avg:.2f} / 5.0 (n={len(scores)})")

    # 결과 저장
    output_file = os.path.join(args.output_dir, "eval_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "total_samples": len(results),
                "avg_score": sum(total_scores) / len(total_scores),
                "format_compliance": format_ok_count / len(results),
                "command_relevance": relevant_count / len(results),
                "category_scores": {
                    k: sum(v) / len(v) for k, v in category_scores.items()
                },
                "detailed_results": results,
            },
            f,
            indent=2,
        )
    print(f"\n결과 저장: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LFM2 Terminal Agent 평가")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/root/outputs/eval")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--use_unsloth", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
