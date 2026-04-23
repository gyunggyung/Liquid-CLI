# Liquid-CLI 🖥️

[한국어 문서](#korean) | [English Document](#english)

---

<a name="korean"></a>
## 🇰🇷 한국어 (Korean)

Liquid AI의 LFM2-8B 모델을 활용한 로컬 터미널 에이전트 CLI입니다. 설정 없이 바로 실행하세요.

### 🎯 목표
복잡한 설정 없이 `pip install` 또는 `uv run`만으로 LFM2 터미널 에이전트의 성능을 로컬에서 체감하는 것을 목표로 합니다.

### ⚠️ 필수 요구 사항
- **리눅스(Linux) 환경**: 이 에이전트는 리눅스 명령어를 생성하고 실행하도록 학습되었습니다.
- **윈도우(Windows) 사용자**: 반드시 **WSL2 (Ubuntu)** 환경에서 사용해 주세요. 윈도우 기본 터미널(CMD/PS)은 지원하지 않습니다.
- **의존성**: `uv`가 설치되어 있어야 합니다.

### 🚀 빠른 시작 (Quick Start)
```bash
cd liquid-cli
uv run liquid ask "영어 지시어 입력 (예: Check current directory)"
```

> [!IMPORTANT]
> **언어 주의**: 모델이 영어 터미널 데이터를 기반으로 학습되었으므로, 지시 사항(Prompt)은 반드시 **영어**로 입력해야 최상의 결과가 나옵니다.

### 🛠️ 사용 예시 (Usage Examples)
1. **파일 및 디렉토리 작업**
   ```bash
   uv run liquid ask "Check what files are in the current folder"
   ```
2. **데이터 검색 및 필터링**
   ```bash
   uv run liquid ask "Find all .json files and extract lines containing '5e8' from them."
   ```
3. **시스템 리소스 확인**
   ```bash
   uv run liquid ask "Check current CPU and Memory usage"
   ```
4. **수동 모델 다운로드**
   ```bash
   uv run liquid setup
   ```

### 📦 주요 기능 (Key Features)
- **Zero-Config**: 첫 실행 시 자동으로 모델을 다운로드하고 캐싱합니다.
- **Agentic Execution**: 상황 분석, 계획 수립 후 최적의 명령어를 제안합니다.
- **Interactive Approval**: 사용자가 승인(`y`)한 명령어만 실제 터미널에서 실행합니다.
- **CPU Optimized**: `llama.cpp` 기반으로 일반 노트북 CPU에서도 빠르게 동작합니다.

### ✅ 장점 및 한계 (Capabilities & Limitations)
- **잘하는 것 (Strengths)**:
  - 시스템 리소스 모니터링 (CPU, RAM 사용량 확인 등)
  - 간단한 파일 조회 및 내용 확인 (ls, cat, 기본 grep)
  - 리눅스 터미널 환경에 최적화된 명령어 생성
- **못하는 것 (Limitations)**:
  - 복잡한 다단계 작업 (현재는 '원샷' 방식이라 탐색 후 주춤할 수 있음)
  - 윈도우 네이티브 명령어 (PowerShell/CMD 지원 안 함)
  - 오타가 섞인 아주 복잡한 지시 (추론이 빗나갈 수 있음)

### 🗺️ 미래 로드맵 (Roadmap)
현재 버전은 한 번의 질문에 답변하는 '원샷' 방식입니다. 향후 다음과 같이 진화합니다:
- **자율 주행 루프 (Agentic Loop)**: 목표 달성까지 에이전트가 스스로 판단하고 행동을 이어갑니다.
- **시스템 인지 능력**: 현재 OS와 설치된 환경을 미리 파악해 헛발질 없는 명령어를 제안합니다.
- **지능적 필터링**: `.venv`, `.git` 등 불필요한 데이터를 자동으로 검색에서 제외합니다.

---

<a name="english"></a>
## 🇺🇸 English

Terminal Agent powered by Liquid AI's LFM2-8B. No setup, just run.

### 🎯 Goal
Experience the power of the LFM2 terminal agent locally with zero configuration, using just `pip install` or `uv run`.

### ⚠️ Requirements
- **Linux Environment**: This agent is optimized for Linux-specific command generation and execution.
- **Windows Users**: You must use **WSL2 (Ubuntu)**. Native Windows terminals (CMD/PS) are not supported.
- **Dependency**: `uv` must be installed.

### 🚀 Quick Start
```bash
cd liquid-cli
uv run liquid ask "Your question in English"
```

> [!IMPORTANT]
> **Language Note**: Please provide instructions in **English** for the best performance.

### 🛠️ Usage Examples
1. **File & Directory Operations**
   ```bash
   uv run liquid ask "Check what files are in the current folder"
   ```
2. **Search & Filter**
   ```bash
   uv run liquid ask "Find all .log files and extract lines containing 'ERROR'"
   ```
3. **System Status**
   ```bash
   uv run liquid ask "Check current CPU and Memory usage"
   ```
4. **Manual Setup**
   ```bash
   uv run liquid setup
   ```

### 📦 Key Features
- **Zero-Config**: Automatic model download and caching on first run.
- **Agentic Execution**: Solves tasks by analyzing context and planning steps.
- **Interactive Approval**: Safety first. Every command requires user confirmation (`y/N`).
- **CPU Optimized**: Fast inference on standard laptop CPUs via `llama.cpp`.

### ✅ Capabilities & Limitations
- **Strengths**:
  - System resource monitoring (CPU, RAM, usage etc.)
  - Simple file exploration and viewing (ls, cat, basic grep)
  - Optimized command generation for Linux terminal environments.
- **Limitations**:
  - Complex multi-turn tasks (Currently 'One-shot', might stop after exploration).
  - Native Windows commands (PowerShell/CMD not supported).
  - Extremely complex prompts with typos (Reasoning may fail).

### 🗺️ Future Roadmap
The current version operates in a 'One-shot' manner. Future updates include:
- **Agentic Loop**: Autonomous execution where the agent continues until the task is complete.
- **System Awareness**: Pre-detecting OS and environment for better accuracy.
- **Smart Filtering**: Automatically excluding heavy directories like `.venv` or `.git`.

---

## 📄 License
Apache License 2.0
