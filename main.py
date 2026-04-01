import argparse

import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.mnist_loader import build_mnist_dataloaders
from src.models.hybrid_pqc import PQCHead, ResNetFeatureExtractor
from src.models.resnet_baseline import build_resnet18_baseline
from src.training.engine import (
    count_trainable_params,
    evaluate,
    print_comparison_table,
    print_history_summary,
    train_model,
)
from src.training.feature_cache import build_feature_cache
from src.utils.seed import get_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNIST 분류: ResNet18 Baseline vs ResNet18 Feature Extractor + PQC"
    )

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--epochs-baseline", type=int, default=3)
    parser.add_argument("--lr-baseline", type=float, default=1e-3)

    parser.add_argument("--epochs-hybrid", type=int, default=3)
    parser.add_argument("--lr-hybrid", type=float, default=5e-3)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--q-layers", type=int, default=2)

    return parser.parse_args()


def make_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        data_dir=args.data_dir,
        seed=args.seed,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        epochs_baseline=args.epochs_baseline,
        lr_baseline=args.lr_baseline,
        epochs_hybrid=args.epochs_hybrid,
        lr_hybrid=args.lr_hybrid,
        n_qubits=args.n_qubits,
        q_layers=args.q_layers,
    )


def main() -> None:
    args = parse_args()
    cfg = make_config(args)

    seed_everything(cfg.seed)
    device = get_device()
    print(f"[정보] device: {device}")

    # 1) 공통 dataloader
    train_loader, val_loader, test_loader = build_mnist_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        val_size=cfg.val_size,
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
    )

    # 2) Baseline 학습
    print("\n[실험 1] ResNet18 Baseline")
    baseline = build_resnet18_baseline(num_classes=10).to(device)
    baseline_hist, baseline_time = train_model(
        baseline,
        train_loader,
        val_loader,
        device,
        cfg.epochs_baseline,
        cfg.lr_baseline,
    )
    baseline_test_loss, baseline_test_acc = evaluate(baseline, test_loader, nn.CrossEntropyLoss(), device)

    # 3) Hybrid 학습
    print("\n[실험 2] ResNet18 Feature Extractor + PQC")
    extractor = ResNetFeatureExtractor().to(device)

    print("[정보] feature cache 생성 중...")
    cached_train = build_feature_cache(extractor, train_loader, device)
    cached_val = build_feature_cache(extractor, val_loader, device)
    cached_test = build_feature_cache(extractor, test_loader, device)

    pin_memory = True if str(device).startswith("cuda") else False
    f_train_loader = DataLoader(
        cached_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    f_val_loader = DataLoader(
        cached_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    f_test_loader = DataLoader(
        cached_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    hybrid = PQCHead(
        in_features=extractor.out_features,
        num_classes=10,
        n_qubits=cfg.n_qubits,
        q_layers=cfg.q_layers,
    ).to(device)

    hybrid_hist, hybrid_time = train_model(
        hybrid,
        f_train_loader,
        f_val_loader,
        device,
        cfg.epochs_hybrid,
        cfg.lr_hybrid,
    )
    hybrid_test_loss, hybrid_test_acc = evaluate(hybrid, f_test_loader, nn.CrossEntropyLoss(), device)

    # 4) 결과 요약
    print_history_summary("ResNet18 Baseline", baseline_hist)
    print_history_summary("ResNet18 Feature + PQC", hybrid_hist)

    results = {
        "ResNet18 Baseline": {
            "test_loss": baseline_test_loss,
            "test_acc": baseline_test_acc,
            "train_time_sec": baseline_time,
            "trainable_params": count_trainable_params(baseline),
        },
        "ResNet18 Feature + PQC": {
            "test_loss": hybrid_test_loss,
            "test_acc": hybrid_test_acc,
            "train_time_sec": hybrid_time,
            "trainable_params": count_trainable_params(hybrid),
        },
    }
    print_comparison_table(results)

    print("\n[해석 가이드]")
    print("- Baseline은 일반적으로 더 높은 정확도를 보일 수 있습니다.")
    print("- Hybrid는 PQC 분류기의 가능성을 확인하고 trainable 파라미터를 줄이는 실험입니다.")
    print("- 본 실험은 noise-free 시뮬레이터(default.qubit) 기준입니다.")


if __name__ == "__main__":
    main()
