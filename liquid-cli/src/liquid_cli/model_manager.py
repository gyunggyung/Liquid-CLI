import os
from huggingface_hub import hf_hub_download
from rich.console import Console

console = Console()

DEFAULT_REPO = "mradermacher/LFM2-8B-Terminal-SFT-Unsloth-GGUF"
DEFAULT_FILE = "LFM2-8B-Terminal-SFT-Unsloth.Q4_K_M.gguf"

class ModelManager:
    def __init__(self, repo_id=DEFAULT_REPO, filename=DEFAULT_FILE):
        self.repo_id = repo_id
        self.filename = filename
        self.local_path = None

    def get_model_path(self):
        """모델 파일의 로컬 경로를 반환하며, 없으면 다운로드 여부를 확인합니다."""
        # 이 메서드는 실제 다운로드를 수행하기보다는 경로를 확인하는 용도로 사용
        # 실제 다운로드는 setup()에서 수행
        try:
            # hf_hub_download는 로컬에 있으면 바로 경로를 반환함 (local_files_only=True)
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_files_only=True
            )
            self.local_path = path
            return path
        except Exception:
            return None

    def download_model(self):
        """모델을 Hugging Face에서 다운로드합니다."""
        console.print(f"[bold blue]Downloading model from {self.repo_id}...[/bold blue]")
        console.print(f"[dim]File: {self.filename}[/dim]")
        
        try:
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                repo_type="model"
            )
            self.local_path = path
            console.print(f"[bold green]✓ Model downloaded to:[/bold green] {path}")
            return path
        except Exception as e:
            console.print(f"[bold red]Error downloading model:[/bold red] {e}")
            return None
