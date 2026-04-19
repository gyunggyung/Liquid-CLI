import subprocess
from rich.console import Console
from rich.prompt import Confirm

console = Console()

class CommandExecutor:
    @staticmethod
    def confirm_and_run(commands):
        """명령어 리스트를 사용자에게 확인받고 실행합니다."""
        if not commands:
            return

        # 명령어 목록 추출
        cmd_list = []
        for cmd in commands:
            if isinstance(cmd, dict):
                cmd_text = cmd.get("keystrokes", "").strip()
            else:
                cmd_text = str(cmd)
            if cmd_text:
                cmd_list.append(cmd_text)

        if not cmd_list:
            return

        # 사용자 확인
        if Confirm.ask("\n[bold cyan]Run these commands?[/bold cyan]"):
            for cmd in cmd_list:
                console.print(f"\n[bold yellow]Executing:[/bold yellow] [white]{cmd}[/white]")
                try:
                    # 실제 명령어 실행
                    # shell=True는 위험할 수 있으나 터미널 에이전트 특성상 필요함
                    result = subprocess.run(
                        cmd, 
                        shell=True, 
                        text=True, 
                        capture_output=True
                    )
                    
                    if result.stdout:
                        console.print("[dim]Output:[/dim]")
                        console.print(result.stdout)
                    if result.stderr:
                        console.print("[bold red]Errors:[/bold red]")
                        console.print(result.stderr)
                        
                except Exception as e:
                    console.print(f"[bold red]Failed to execute command '{cmd}':[/bold red] {e}")
        else:
            console.print("[yellow]Execution cancelled.[/yellow]")
