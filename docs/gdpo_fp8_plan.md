# GDPO FP8 최적화 도입 계획서

본 문서는 Unsloth의 최신 FP8(8-bit Floating Point) Reinforcement Learning 기술을 **Liquid-CLI** 프로젝트의 GDPO 단계에 도입하기 위한 상세 계획을 담고 있습니다.

## 1. 도입 목적
- **메모리 효율 극대화**: H100 GPU에서 모델 가중치 및 옵티마이저 상태를 FP8로 관리하여 메모리 점유율을 약 40~50% 절감.
- **학습 속도 향상**: FP8 하드웨어 가속을 통해 GRPO/GDPO의 핵심인 그룹 샘플링 및 롤아웃 생성 속도 개선.
- **학습 안정성**: Unsloth의 다이내믹 퀀타이제이션을 통해 BF16 대비 높은 지능 수준 유지.

## 2. 하드웨어 및 소프트웨어 전제 조건
- **GPU**: NVIDIA H100 (현재 Vessl 가동 중, FP8 커널 완벽 지원)
- **Library**: 
  - `unsloth` 최신 버전 (2025.x 이상)
  - `vLLM` 0.8.0 이상 (LoRA 지원 버전)
  - LFM2 지원 전용 `transformers` (현재 빌드 설치 완료)

## 3. 세부 구현 계획

### A. 모델 로딩 최적화
- `FastLanguageModel.from_pretrained` 호출 시 `load_in_fp8=True` 옵션 강제 적용.
- `FastLanguageModel.for_training(model)` 호출을 통한 실시간 최적화 커널 주입.

### B. vLLM 호환성 패치
- GDPO의 다중 롤아웃(Rollout) 생성을 위해 `unsloth_zoo`의 최신 패치를 적용:
  ```python
  from unsloth_zoo.patching import patch_vllm_group_logprobs
  patch_vllm_group_logprobs()
  ```
- 이를 통해 vLLM이 FP8 가중치를 사용하면서도 정확한 Logprobs를 계산하도록 보장.

### C. 옵티마이저 상태 최적화
- `bitsandbytes`의 8-bit AdamW 옵티마이저 또는 Unsloth 전용 FP8 옵티마이저 검토.
- `weight_decay` 및 `learning_rate` 스케줄링 최적화.

### D. 데이터 레이아웃
- `UNSLOTH_MOE_BACKEND="grouped_mm"` 설정을 유지하여 MoE 아키텍처의 FP8 연산 효율 극대화.

## 4. VRAM 최적 활용 전략 (Throughput 극대화)
FP8 도입으로 확보된 메모리 여분(약 50GB 이상)을 단순히 비워두지 않고, 학습 효율을 극한으로 끌어올리는 데 집중적으로 투입합니다.

### A. 롤아웃 생성량(num_generations) 극대화
- 현재 기본값 4~8에서 **16~32** 이상으로 상향 조정.
- 한 번의 스텝에서 더 샘플링을 많이 할수록 리워드 신호가 정교해지며, H100의 80GB VRAM을 가장 가치 있게 사용하는 방법입니다.

### B. 로컬 배치 사이즈(micro_batch_size) 튜닝
- H100의 Tensor Core 가동률을 높이기 위해 배치 사이즈를 단계적으로 높입니다.
- **목표 지점**: `nvidia-smi` 기준 VRAM 점유율이 **72GB~76GB(약 90~95%)**에 도달할 때까지 파라미터를 상향 조절하여 GPU 유휴 시간을 최소화합니다.

### C. Gradient Accumulation 최적화
- 배치 사이즈 확대에 따라 `gradient_accumulation_steps`를 조정하여 연산 시간과 서버 간 통신 시간의 균형을 맞춥니다.

## 5. 기대 효과
- **비약적인 샘플 효율성**: 동일한 GPU 시간 동안 훨씬 많은 보상 데이터를 학습하여 모델 성능 향상.
- **최고의 가성비**: 주어진 H100 자원을 100% 가동하여 전체 학습 기간을 단축, Vessl 크레딧 소모 최적화.

## 6. 단계별 실행 로드맵
1. **SFT 완료**: 현재 진행 중인 SFT 학습 및 모델 저장 완료.
2. **라이브러리 최종 점검**: `vllm.lora.models` 모듈 수정을 위한 버전 최적화.
3. **코드 업데이트**: 위 최적화 기법을 반영한 `train_gdpo.py` 최종 배포.
4. **GDPO 실행**: FP8 모드로 강화학습 시작.

## 7. 서버 리셋 대비 모델 로딩 전략
Vessl 서버가 재시작되어 `/root/outputs/sft/final` 경로의 로컬 모델이 사라진 경우에도, 허깅페이스 허브를 통해 즉시 학습을 재개할 수 있습니다.

- **자동 다운로드 기능**: `train_gdpo.py`의 `--model_path` 인자에 로컬 경로 대신 **HuggingFace 리포지토리 ID**를 입력하면 Unsloth가 자동으로 Hub에서 가중치를 내려받습니다.
- **실행 예시**:
  ```bash
  # 로컬 파일이 없을 때 (허브에서 직접 로딩)
  python train_gdpo.py \
      --model_path gyung/LFM2-8B-Terminal-SFT \
      --push_to_hub
  ```
- **주의 사항**: 사전에 `huggingface-cli login`이 완료되어 있어야 본인 소유의 Private 리포지토리에 접근 가능합니다.

---
*문서 작성일: 2026-04-17*
*참고 문서: [Unsloth FP8 RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning)*
