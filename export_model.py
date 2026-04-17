"""
export_model.py — 학습된 모델을 GGUF (INT8/Q4) 포맷으로 변환

Usage:
    python export_model.py --model_path /root/outputs/sft/final
    python export_model.py --model_path /root/outputs/gdpo/merged --quant q4_k_m
"""

import argparse
import os


def main(args):
    print("=" * 60)
    print("모델 변환: GGUF 포맷")
    print("=" * 60)
    print(f"  Model: {args.model_path}")
    print(f"  Quantization: {args.quant}")
    print(f"  Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        trust_remote_code=True,
    )

    # GGUF 변환
    print(f"\n[1/2] GGUF {args.quant} 변환 중...")
    model.save_pretrained_gguf(
        args.output_dir,
        tokenizer,
        quantization_method=args.quant,
    )

    # 파일 크기 확인
    print(f"\n[2/2] 변환 결과:")
    for f in os.listdir(args.output_dir):
        fpath = os.path.join(args.output_dir, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            size_gb = size_mb / 1024
            if size_gb >= 1:
                print(f"  {f}: {size_gb:.2f} GB")
            else:
                print(f"  {f}: {size_mb:.1f} MB")

    print(f"\n✅ 변환 완료!")
    print(f"\nllama.cpp로 실행:")
    print(
        f"  ./llama-cli -m {args.output_dir}/*.gguf "
        f'-p "You are a terminal agent..." '
        f"--temp 0.3 --top-k 50 -n 512 -t 8"
    )

    if args.push_to_hub:
        print(f"\n[선택] HuggingFace 업로드:")
        model.push_to_hub_gguf(
            args.hub_repo,
            tokenizer,
            quantization_method=args.quant,
            token=os.environ.get("HF_TOKEN", ""),
        )
        print(f"  ✅ 업로드: {args.hub_repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GGUF 모델 변환")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/root/outputs/gguf")
    parser.add_argument(
        "--quant",
        type=str,
        default="q8_0",
        choices=["q8_0", "q4_k_m", "q5_k_m", "q6_k", "f16"],
    )
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--hub_repo", type=str, default="gyung/LFM2-8B-Terminal")
    args = parser.parse_args()
    main(args)
