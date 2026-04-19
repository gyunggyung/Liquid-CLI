# [Liquid-CLI] Python CLI 애플리케이션 'liquid' 구현 계획

이 계획은 사용자가 `pip install`만으로 Liquid AI의 LFM2 터미널 에이전트 모델을 로컬(CPU)에서 쉽게 실행하고 상호작용할 수 있는 CLI 도구를 만드는 것을 목표로 합니다.

## User Review Required

> [!IMPORTANT]
> **대화형 명령어 실행 모델**: 터미널 에이전트 특성상 모델이 제안한 리눅스 명령어를 사용자가 수락 시 실제 실행하는 기능을 포함할 예정입니다. 보안을 위해 **모든 명령어에 대해 사용자 확인(`y/N`)** 단계를 거치도록 설계했습니다.

> [!NOTE]
> **Zero-Config용 테스트 모델**: 현재 사용자가 제안한 `mradermacher/LFM2-8B-Terminal-SFT-Unsloth-GGUF` 레포지토리의 `LFM2-8B-Terminal-SFT-Unsloth.Q4_K_M.gguf` 파일을 기본 테스트 모델로 사용합니다. 향후 사용자의 자체 GGUF로 교체하기 쉽게 설정 파일이나 상수로 관리합니다.

## Proposed Changes

새로운 폴더 `liquid-cli`를 생성하고 `uv`를 활용한 배포 가능한 패키지 구조를 구축합니다.

### 1. 인프라 및 설정 [NEW]

#### [NEW] [pyproject.toml](file:///c:/github/Liquid-CLI/liquid-cli/pyproject.toml)
- `uv` 기반의 의존성 관리 설정.
- `llama-cpp-python`, `huggingface_hub` (모델 다운로드용), `rich` (UI), `typer` (CLI), `prompt_toolkit` (입력) 등 포함.

---

### 2. 코어 로직 (`src/liquid_cli/`) [NEW]

#### [NEW] [main.py](file:///c:/github/Liquid-CLI/liquid-cli/src/liquid_cli/main.py)
- CLI 엔트리포인트 (Typer 사용).
- `ask` (또는 `search` 별칭), `chat`, `setup` 명령어 정의.
- **Zero-Config 로직**: 실행 시 모델이 없으면 자동으로 `setup`을 호출하여 다운로드 진행.

#### [NEW] [engine.py](file:///c:/github/Liquid-CLI/liquid-cli/src/liquid_cli/engine.py)
- `llama-cpp-python` 래퍼 클래스.
- 터미널 에이전트 특화 프롬프트 템플릿 처리.

#### [NEW] [model_manager.py](file:///c:/github/Liquid-CLI/liquid-cli/src/liquid_cli/model_manager.py)
- `hf_hub_download`를 이용한 모델 자동 다운로드 및 캐싱 로직.
- 기본값: `mradermacher/LFM2-8B-Terminal-SFT-Unsloth-GGUF/LFM2-8B-Terminal-SFT-Unsloth.Q4_K_M.gguf`

#### [NEW] [ui.py](file:///c:/github/Liquid-CLI/liquid-cli/src/liquid_cli/ui.py)
- `Rich`를 이용한 터미널 UI 구성.
- 에이전트의 JSON 응답(Analysis, Plan, Commands)을 패널로 구분하여 렌더링.

---

### 3. 기능성 유틸리티 [NEW]

#### [NEW] [executor.py](file:///c:/github/Liquid-CLI/liquid-cli/src/liquid_cli/executor.py)
- 모델이 제안한 명령어를 안전하게 실행하는 로직.
- 사용자 승인 인터페이스 (`Confirm.ask`).

## Open Questions

- **명령어 자동 실행 범위**: 안전을 위해 일단 **모든 명령어를 확인 후 실행**하는 방식으로 진행하며, 추후 설정으로 변경 가능하게 하겠습니다.
- **별칭 지원**: `liquid search`도 `liquid ask`와 동일하게 작동하도록 처리하겠습니다.
