from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live
from rich.spinner import Spinner

console = Console()

class LiquidUI:
    @staticmethod
    def show_welcome():
        console.print(Panel.fit(
            "[bold cyan]Liquid-CLI[/bold cyan] - LFM2 Terminal Agent",
            subtitle="[dim]Powered by Liquid AI & llama.cpp[/dim]"
        ))

    @staticmethod
    def show_response(response):
        """에이전트의 JSON 응답을 보기 좋게 출력합니다."""
        
        # 1. Analysis
        if "analysis" in response:
            console.print(Panel(
                Markdown(response["analysis"]),
                title="[bold yellow]Analysis[/bold yellow]",
                border_style="yellow"
            ))

        # 2. Plan
        if "plan" in response:
            console.print(Panel(
                Markdown(response["plan"]),
                title="[bold green]Plan[/bold green]",
                border_style="green"
            ))

        # 3. Commands
        if "commands" in response and response["commands"]:
            table = Table(title="Suggested Commands", show_header=True, header_style="bold magenta")
            table.add_column("Type", style="dim")
            table.add_column("Command")
            
            for cmd in response["commands"]:
                if isinstance(cmd, dict):
                    # Nemotron format: {"keystrokes": "ls\n"}
                    raw_cmd = cmd.get("keystrokes", "").strip()
                else:
                    raw_cmd = str(cmd)
                
                table.add_row("Shell", Syntax(raw_cmd, "bash", theme="monokai"))
            
            console.print(table)

    @staticmethod
    def spinner(message="Thinking..."):
        return Live(Spinner("dots", text=message, style="cyan"), transient=True)
