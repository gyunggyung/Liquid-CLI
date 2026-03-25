"""
prepare_data.py — Nemotron-Terminal-Corpus 비코딩 서브셋 필터링

Usage:
    python prepare_data.py [--output_dir /root/data] [--max_seq_length 8192]
"""

import argparse
import json
import os

from datasets import concatenate_datasets, load_dataset


# ═══ 코딩 의존도가 높은 키워드 (필터링용) ═══
CODING_KEYWORDS = [
    # Python 코드 작성 관련
    "write a python", "write python", "implement a function", "implement the",
    "solution.py", "def ", "class ", "import pandas", "import numpy",
    "import sklearn", "import scipy", "import matplotlib", "import tensorflow",
    "import torch", "from torch", "import keras",
    # 코딩 태스크 지시어
    "write code", "write a script", "write a program",
    "debug the code", "fix the bug", "fix the error in",
    "refactor", "optimize the code", "implement algorithm",
    # 소프트웨어 엔지니어링
    "solution.patch", "git diff", "pull request",
    "graph traversal", "binary search", "dynamic programming",
    "linked list", "binary tree", "hash map",
    # 데이터 사이언스 (Python 의존)
    "train a model", "fit the model", "neural network",
    "regression", "classification model",
    "pandas dataframe", "scikit-learn",
]

# ═══ 비코딩 도메인 (domain 필드가 있을 경우 사용) ═══
NON_CODING_DOMAINS = [
    "file_operations",
    "data_processing",
    "data_querying",
    "dependency_management",
    "security",
]

EXCLUDED_DOMAINS = [
    "data_science",
    "scientific_computing",
    "debugging",
    "software_engineering",
]


def has_coding_keywords(text: str) -> bool:
    """텍스트에 코딩 관련 키워드가 포함되어 있는지 확인"""
    text_lower = text.lower()
    return any(kw in text_lower for kw in CODING_KEYWORDS)


def extract_text_from_example(example: dict) -> str:
    """데이터셋 예시에서 필터링 대상 텍스트 추출"""
    parts = []
    for key in ["instruction", "prompt", "input", "task", "messages", "text"]:
        val = example.get(key)
        if val is not None:
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        parts.append(str(item.get("content", "")))
                    else:
                        parts.append(str(item))
            else:
                parts.append(str(val))
    return " ".join(parts)


def filter_non_coding(example: dict) -> bool:
    """비코딩 데이터인지 판별"""
    # 1) domain 필드가 있으면 우선 사용
    domain = example.get("domain", None)
    if domain is not None:
        if domain in EXCLUDED_DOMAINS:
            return False
        if domain in NON_CODING_DOMAINS:
            return True
        # 알 수 없는 domain은 키워드 필터링으로 넘어감

    # 2) 키워드 기반 필터링
    text = extract_text_from_example(example)
    return not has_coding_keywords(text)


def main(args):
    print("=" * 60)
    print("Nemotron-Terminal-Corpus 데이터 준비")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Step 1: skill_based 스플릿 로딩 ───
    # dataset_adapters는 Math/Code/SWE → 전부 제외
    print("\n[1/4] skill_based 스플릿 로딩...")
    splits = {}
    for split_name in ["skill_based_easy", "skill_based_medium", "skill_based_mixed"]:
        print(f"  Loading {split_name}...")
        try:
            ds = load_dataset(
                "nvidia/Nemotron-Terminal-Corpus", split_name, split="train"
            )
            splits[split_name] = ds
            print(f"    → {len(ds)} samples, columns: {ds.column_names}")
        except Exception as e:
            print(f"    ⚠️ Failed to load {split_name}: {e}")

    if not splits:
        print("❌ 데이터를 로딩할 수 없습니다.")
        return

    all_data = concatenate_datasets(list(splits.values()))
    print(f"\n  전체 skill_based: {len(all_data)} samples")

    # ─── Step 2: 비코딩 필터링 ───
    print("\n[2/4] 비코딩 데이터 필터링...")
    filtered = all_data.filter(
        filter_non_coding,
        num_proc=4,
        desc="Filtering non-coding",
    )
    print(f"  필터링 전: {len(all_data)} → 필터링 후: {len(filtered)}")
    print(f"  제거된 비율: {1 - len(filtered)/len(all_data):.1%}")

    # ─── Step 3: 도메인 분포 출력 ───
    if "domain" in filtered.column_names:
        print("\n[3/4] 도메인 분포:")
        from collections import Counter

        domain_counts = Counter(filtered["domain"])
        for domain, count in domain_counts.most_common():
            print(f"  {domain}: {count} ({count/len(filtered):.1%})")
    else:
        print("\n[3/4] domain 필드 없음 — 키워드 기반 필터링만 적용됨")

    # ─── Step 4: 저장 ───
    print(f"\n[4/4] 저장: {args.output_dir}")

    # SFT용 데이터 저장
    sft_path = os.path.join(args.output_dir, "sft_data")
    filtered.save_to_disk(sft_path)
    print(f"  SFT 데이터 저장: {sft_path} ({len(filtered)} samples)")

    # RLVR용 데이터 (prompt/answer 포맷)
    # 실제 필드 구조에 따라 조정 필요
    print(f"\n  데이터 필드 확인 (첫 번째 샘플):")
    sample = filtered[0]
    for k, v in sample.items():
        val_str = str(v)[:200] if len(str(v)) > 200 else str(v)
        print(f"    {k}: {val_str}")

    print("\n✅ 데이터 준비 완료!")
    print(f"\n다음 단계: python train_sft.py --data_path {sft_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nemotron-Terminal-Corpus 데이터 준비")
    parser.add_argument(
        "--output_dir", type=str, default="/root/data", help="출력 디렉토리"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=8192, help="최대 시퀀스 길이"
    )
    args = parser.parse_args()
    main(args)
