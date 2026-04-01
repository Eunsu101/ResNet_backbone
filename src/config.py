from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    # 데이터 관련 설정
    data_dir: str = "./data"
    image_size: int = 96
    batch_size: int = 128
    val_size: int = 5000
    num_workers: int = 2

    # 재현성
    seed: int = 42

    # Baseline 설정
    epochs_baseline: int = 3
    lr_baseline: float = 1e-3

    # Hybrid 설정
    epochs_hybrid: int = 3
    lr_hybrid: float = 5e-3
    n_qubits: int = 4
    q_layers: int = 2
