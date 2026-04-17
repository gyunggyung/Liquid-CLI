from datasets import load_from_disk
import os

def upload_to_hub():
    # 1. 데이터 경로 설정
    dataset_path = "/root/data/sft_data"
    repo_id = "gyung/LFM2-Terminal-SFT-Processed" # 원하는 리포지토리 이름으로 변경 가능

    print(f"🚀 로컬 데이터 로드 중: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"❌ 에러: {dataset_path} 경로가 존재하지 않습니다.")
        return

    dataset = load_from_disk(dataset_path)
    print(f"✅ 데이터 로드 완료 ({len(dataset)} samples)")

    # 2. 업로드 시작
    print(f"📤 HuggingFace Hub 업로드 시작: {repo_id}")
    try:
        # 데이터셋 뷰어 활용을 위해 Public으로 업로드합니다.
        dataset.push_to_hub(repo_id, private=False) 
        print(f"🎉 업로드 성공! 주소: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"❌ 업로드 중 에러 발생: {e}")
        print("💡 팁: 'huggingface-cli login'이 되어있는지 확인하세요.")

if __name__ == "__main__":
    upload_to_hub()
