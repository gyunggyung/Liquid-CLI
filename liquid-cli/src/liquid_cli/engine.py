import json
from llama_cpp import Llama
from rich.console import Console

console = Console()

SYSTEM_PROMPT = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {"keystrokes": "ls -la\\n", "duration": 0.1},
    {"keystrokes": "cd project\\n", "duration": 0.1}
  ],
  "task_complete": false
}"""

class InferenceEngine:
    def __init__(self, model_path, n_ctx=4096, n_threads=8):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.llm = None

    def load(self):
        """모델을 메모리에 로드합니다."""
        if self.llm is None:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False
            )

    def ask(self, query, history=None):
        """단일 질문에 대해 에이전트 응답을 생성합니다."""
        self.load()
        
        # 기본 대화 구성
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {query}"}
        ]
        
        # 추론 수행
        response = self.llm.create_chat_completion(
            messages=messages,
            response_format={
                "type": "json_object"
            },
            temperature=0.3,
        )
        
        content = response["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 보수적 처리: JSON 파싱 실패 시 원문 반환 시도 (문자열 처리 로직 필요할 수 있음)
            return {"analysis": "Error parsing response", "plan": "Raw output: " + content, "commands": []}
