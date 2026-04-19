import typer
from typing import Optional
from .model_manager import ModelManager
from .engine import InferenceEngine
from .ui import LiquidUI
from .executor import CommandExecutor
from rich.console import Console
from rich.prompt import Confirm

app = typer.Typer(help="Liquid-CLI: LFM2 Terminal Agent")
console = Console()

def ensure_model():
    """모델이 로컬에 있는지 확인하고 없으면 다운로드를 제안합니다."""
    mm = ModelManager()
    path = mm.get_model_path()
    
    if path:
        return path
        
    console.print("[yellow]Model not found locally.[/yellow]")
    if Confirm.ask("[bold cyan]Would you like to download the LFM2-8B model from Hugging Face?[/bold cyan]"):
        path = mm.download_model()
        return path
    else:
        console.print("[red]Model is required to run Liquid-CLI.[/red]")
        raise typer.Exit()

@app.command()
def ask(
    query: str = typer.Argument(..., help="The task or question for the terminal agent"),
    context_length: int = typer.Option(4096, "--ctx", "-c", help="Context window length")
):
    """Ask the terminal agent to perform a task."""
    LiquidUI.show_welcome()
    
    # 1. Ensure model exists
    model_path = ensure_model()
    
    # 2. Initialize engine
    engine = InferenceEngine(model_path, n_ctx=context_length)
    
    # 3. Predict
    with LiquidUI.spinner(f"Agent is thinking about: {query}"):
        response = engine.ask(query)
    
    # 4. Show UI
    LiquidUI.show_response(response)
    
    # 5. Execute if needed
    if "commands" in response and response["commands"]:
        CommandExecutor.confirm_and_run(response["commands"])

@app.command()
def search(
    query: str = typer.Argument(..., help="Alias for 'ask'"),
    context_length: int = typer.Option(4096, "--ctx", "-c")
):
    """Alias for 'ask' command."""
    ask(query, context_length)

@app.command()
def setup():
    """Download the model manually."""
    mm = ModelManager()
    mm.download_model()

if __name__ == "__main__":
    app()
