"""
evaluate_vllm.py — vLLM을 사용한 고성능 LFM2 터미널 에이전트 평가 스크립트

Usage:
    python evaluate_vllm.py --model_path /root/outputs/sft_unsloth/checkpoint-1000
"""

import argparse
import json
import os
import time
from vllm import LLM, SamplingParams

# evaluate.py에서 프롬프트 공유
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
    """단일 응답 평가 (evaluate.py와 동일 로직)"""
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

    try:
        parsed = json.loads(response)
        result["json_valid"] = True
        result["score"] += 1.0
    except (json.JSONDecodeError, TypeError):
        try:
            json_match = response[response.index("{") : response.rindex("}") + 1]
            parsed = json.loads(json_match)
            result["json_valid"] = True
            result["score"] += 0.5
        except (ValueError, json.JSONDecodeError):
            return result

    if "analysis" in parsed and parsed["analysis"]:
        result["has_analysis"] = True
        result["score"] += 1.0
    if "plan" in parsed and parsed["plan"]:
        result["has_plan"] = True
        result["score"] += 1.0
    if "commands" in parsed and isinstance(parsed["commands"], list) and parsed["commands"]:
        result["has_commands"] = True
        result["score"] += 1.0

        for cmd in parsed["commands"]:
            ks = cmd.get("keystrokes", "").strip()
            if ks:
                words = ks.split()
                if not words: continue
                first_word = words[0].rstrip("\n")
                if first_word == "sudo" and len(words) > 1:
                    first_word = words[1].rstrip("\n")
                result["command_names"].append(first_word)

        for expected in expected_commands:
            if expected in result["command_names"]:
                result["commands_relevant"] = True
                result["score"] += 1.0
                break

    result["format_ok"] = all([result["json_valid"], result["has_analysis"], result["has_plan"], result["has_commands"]])
    return result

def main(args):
    print("=" * 60)
    print("LFM2 Terminal Agent 고속 평가 (vLLM)")
    print("=" * 60)

    # vLLM 엔진 초기화
    # ⚠️ 학습 중 실행 시 gpu_memory_utilization을 낮춰야 합니다.
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_tokens=1024,
    )

    # 1. 프롬프트 배치 생성
    prompts = []
    # LFM2 Chat Template 수동 구성
    for test in EVAL_PROMPTS:
        prompt_text = (
            f"<|system|>\n{SYSTEM_PROMPT}<|end_of_text|>\n"
            f"<|user|>\nCurrent terminal state:\n$ \n\nTask: {test['prompt']}<|end_of_text|>\n"
            f"<|assistant|>\n"
        )
        prompts.append(prompt_text)

    # 2. 배치 추론 (vLLM 핵심)
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time

    # 3. 결과 집계 및 평가
    results = []
    category_scores = {}

    for i, output in enumerate(outputs):
        test = EVAL_PROMPTS[i]
        response = output.outputs[0].text
        
        eval_result = evaluate_response(response, test["expected_commands"])
        eval_result["id"] = test["id"]
        eval_result["category"] = test["category"]
        results.append(eval_result)

        cat = test["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(eval_result["score"])

        status = "✅" if eval_result["format_ok"] else "❌"
        print(f"  {status} [{test['id']}] score={eval_result['score']:.1f} cmds={eval_result['command_names']}")

    # 4. 요약 출력
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print("\n" + "=" * 60)
    print("평가 결과 요약 (vLLM)")
    print("=" * 60)
    print(f"소요 시간: {total_time:.2f}초")
    print(f"평균 속도: {total_tokens/total_time:.2f} tokens/s")
    
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"전체 평균 점수: {avg_score:.2f} / 5.0")

    # 결과 저장
    output_file = os.path.join(args.output_dir, "eval_vllm_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({"detailed_results": results, "summary": {"avg_score": avg_score, "tps": total_tokens/total_time}}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LFM2 Terminal Agent vLLM 평가")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--output_dir", type=str, default="/root/outputs/eval_vllm")
    args = parser.parse_args()
    main(args)
