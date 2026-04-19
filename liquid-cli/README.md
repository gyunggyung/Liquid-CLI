# Liquid-CLI 🖥️

[한국어 문서](#korean) | [English Document](#english)

---

<a name="korean"></a>
## 🇰🇷 한국어 (Korean)

Liquid AI의 LFM2-8B 모델을 활용한 로컬 터미널 에이전트 CLI입니다.

### 🎯 목표
복잡한 설정 없이 `pip install` 또는 `uv run`만으로 LFM2 터미널 에이전트의 성능을 로컬에서 체감하는 것을 목표로 합니다.

### ⚠️ 필수 요구 사항
- **리눅스(Linux) 환경**: 이 에이전트는 리눅스 명령어를 생성하고 실행하도록 학습되었습니다.
- **윈도우(Windows) 사용자**: 반드시 **WSL2 (Ubuntu)** 환경에서 사용해 주세요. 윈도우 기본 터미널(CMD/PS)은 지원하지 않습니다.
- **의존성**: `uv`가 설치되어 있어야 합니다.

### 🚀 빠른 시작
`uv`가 설치되어 있다면 설치 과정 없이 아래 명령어로 바로 실행할 수 있습니다.

```bash
cd liquid-cli
uv run liquid ask "명령어는 영어로 입력해 주세요 (예: Check current directory)"
```

> [!IMPORTANT]
> **중요**: 모델이 영어 터미널 데이터를 기반으로 학습되었으므로, 지시 사항(Prompt)은 반드시 **영어**로 입력해야 최상의 결과를 얻을 수 있습니다.

### 🛠️ 사용 예시
1. **파일 및 디렉토리 확인**
   ```bash
   uv run liquid ask "Check what files are in the current folder"
   ```
2. **데이터 필터링**
   ```bash
   uv run liquid ask "Find all .log files and extract lines containing 'ERROR'"
   ```
3. **시스템 상태 확인**
   ```bash
   uv run liquid ask "Check current CPU and Memory usage"
   ```

---

<a name="english"></a>
## 🇺🇸 English

A local terminal agent CLI using Liquid AI's LFM2-8B model.

### 🎯 Goal
Experience the power of the LFM2 terminal agent locally with zero configuration, using just `pip install` or `uv run`.

### ⚠️ Requirements
- **Linux Environment**: This agent is trained to generate and execute Linux-specific commands.
- **Windows Users**: You must use **WSL2 (Ubuntu)**. Native Windows terminals (CMD/PS) are not supported.
- **Dependency**: `uv` must be installed.

### 🚀 Quick Start
If you have `uv` installed, you can run the agent directly without any setup.

```bash
cd liquid-cli
uv run liquid ask "Your question in English"
```

> [!IMPORTANT]
> **Note**: Since the model is trained on English terminal corpora, please provide instructions in **English** for the best performance.

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

---

## 📦 Key Features / 주요 기능
- **Zero-Config**: Automatic model download on first run.
- **Agentic Execution**: Solves terminal tasks by analyzing, planning, and executing.
- **Interactive Approval**: Executes commands only after user confirmation.
- **CPU Optimized**: Fast inference on standard laptop CPUs.

## 📄 License
Apache License 2.0
