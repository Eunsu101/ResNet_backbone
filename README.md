# MNIST: ResNet18 vs ResNet18 Feature Extractor + PQC

이 프로젝트는 MNIST 분류에서 아래 두 모델을 동일 조건으로 비교합니다.

1. ResNet18 Baseline
2. ResNet18 Feature Extractor(고정) + PennyLane PQC 분류기

양자 회로는 노이즈를 고려하지 않는 noise-free 시뮬레이터(`default.qubit`)를 사용합니다.

## 폴더 구조

- `main.py`: 실험 실행 진입점
- `src/config.py`: 실험 설정
- `src/data/mnist_loader.py`: MNIST dataloader
- `src/models/resnet_baseline.py`: Baseline ResNet18
- `src/models/hybrid_pqc.py`: Feature extractor + PQC 헤드
- `src/training/engine.py`: 학습/평가 루프 및 비교 출력
- `src/training/feature_cache.py`: feature caching
- `src/utils/seed.py`: 시드/디바이스 유틸

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
python main.py
```

## 주요 인자 예시

```bash
python main.py --epochs-baseline 3 --epochs-hybrid 3 --n-qubits 4 --q-layers 2
```
