import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.mnist_loader import build_mnist_dataloaders, build_mnist_oneclass_dataloaders
from src.models.hybrid_pqc import PQCHead, ResNetFeatureExtractor
from src.models.resnet_baseline import build_resnet18_baseline
from src.training.anomaly_eval import (
    compute_binary_metrics,
    compute_euclidean_scores,
    compute_mahalanobis_scores,
    fit_gaussian_from_normal,
    plot_roc_pr,
    plot_score_histogram,
    plot_tsne_umap,
    threshold_from_normal,
)
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

    parser.add_argument("--oneclass-eval", action="store_true")
    parser.add_argument("--normal-digit", type=int, default=0)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--viz-max-samples", type=int, default=3000)
    parser.add_argument("--output-dir", type=str, default="./artifacts")

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

    if args.oneclass_eval:
        run_oneclass_eval(cfg, args, device)
        return

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
    baseline = build_resnet18_baseline(num_classes=10, pretrained=not args.scratch).to(device)
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


class FrozenResNetExtractor(nn.Module):
    """학습 완료된 ResNet18에서 fc를 제거해 feature를 추출한다."""

    def __init__(self, trained_model: nn.Module):
        super().__init__()
        self.features = nn.Sequential(*list(trained_model.children())[:-1])
        self.out_features = trained_model.fc.in_features
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.features(x)
            feat = torch.flatten(feat, 1)
        return feat


def run_oneclass_eval(cfg: ExperimentConfig, args: argparse.Namespace, device: torch.device) -> None:
    print("\n[실험] Scratch ResNet Feature Extractor One-Class 평가")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A) Scratch feature pretext 학습(10-class)
    train_loader, val_loader, test_loader = build_mnist_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        val_size=cfg.val_size,
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
    )

    feature_backbone = build_resnet18_baseline(num_classes=10, pretrained=False).to(device)
    print("[정보] scratch ResNet pretext 학습 시작")
    _, pretext_time = train_model(
        feature_backbone,
        train_loader,
        val_loader,
        device,
        cfg.epochs_baseline,
        cfg.lr_baseline,
    )
    pretext_test_loss, pretext_test_acc = evaluate(feature_backbone, test_loader, nn.CrossEntropyLoss(), device)

    # B) One-class split 생성(train=정상만)
    oc_train_loader, oc_val_loader, oc_test_loader = build_mnist_oneclass_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        val_size=cfg.val_size,
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        normal_digit=args.normal_digit,
    )

    extractor = FrozenResNetExtractor(feature_backbone).to(device)
    print("[정보] one-class feature cache 생성 중...")
    train_cache = build_feature_cache(extractor, oc_train_loader, device)
    val_cache = build_feature_cache(extractor, oc_val_loader, device)
    test_cache = build_feature_cache(extractor, oc_test_loader, device)

    train_feats, _ = train_cache.tensors
    val_feats, val_labels = val_cache.tensors
    test_feats, test_labels = test_cache.tensors

    # C) 점수화 방식 1: Euclidean to center
    center = train_feats.mean(dim=0)
    val_scores_euc = compute_euclidean_scores(val_feats, center)
    test_scores_euc = compute_euclidean_scores(test_feats, center)

    val_labels_np = val_labels.numpy()
    test_labels_np = test_labels.numpy()

    thr_euc = threshold_from_normal(val_scores_euc[val_labels_np == 0], quantile=0.95)
    metrics_euc = compute_binary_metrics(test_labels_np, test_scores_euc, thr_euc)

    # D) 점수화 방식 2: Mahalanobis
    mean_g, cov_inv = fit_gaussian_from_normal(train_feats)
    val_scores_mah = compute_mahalanobis_scores(val_feats, mean_g, cov_inv)
    test_scores_mah = compute_mahalanobis_scores(test_feats, mean_g, cov_inv)

    thr_mah = threshold_from_normal(val_scores_mah[val_labels_np == 0], quantile=0.95)
    metrics_mah = compute_binary_metrics(test_labels_np, test_scores_mah, thr_mah)

    # E) 시각화
    max_samples = min(args.viz_max_samples, test_feats.shape[0])
    viz_feats = test_feats[:max_samples].numpy()
    viz_labels = test_labels_np[:max_samples]

    plot_tsne_umap(
        viz_feats,
        viz_labels,
        out_path=out_dir / "embedding_tsne_umap.png",
        seed=cfg.seed,
        title="Scratch ResNet Feature Embedding (normal=0, anomaly=1-9)",
    )
    plot_roc_pr(
        test_labels_np,
        test_scores_mah,
        out_path=out_dir / "roc_pr_mahalanobis.png",
        title_prefix="Mahalanobis",
    )
    plot_score_histogram(
        test_labels_np,
        test_scores_mah,
        threshold=thr_mah,
        out_path=out_dir / "score_hist_mahalanobis.png",
        title="Mahalanobis score distribution",
    )

    print("\n[Pretext 10-class 성능]")
    print(f"- Test Loss: {pretext_test_loss:.4f}")
    print(f"- Test Acc : {pretext_test_acc:.4f}")
    print(f"- Train Time(s): {pretext_time:.2f}")

    print("\n[One-class 결과] Euclidean")
    print(
        f"- AUROC: {metrics_euc['auroc']:.4f}, AUPRC: {metrics_euc['auprc']:.4f}, "
        f"FPR95: {metrics_euc['fpr95']:.4f}, TPR@FPR5%: {metrics_euc['tpr_at_fpr05']:.4f}"
    )
    print(
        f"- Confusion (TN/FP/FN/TP): {int(metrics_euc['tn'])}/{int(metrics_euc['fp'])}/"
        f"{int(metrics_euc['fn'])}/{int(metrics_euc['tp'])}"
    )

    print("\n[One-class 결과] Mahalanobis")
    print(
        f"- AUROC: {metrics_mah['auroc']:.4f}, AUPRC: {metrics_mah['auprc']:.4f}, "
        f"FPR95: {metrics_mah['fpr95']:.4f}, TPR@FPR5%: {metrics_mah['tpr_at_fpr05']:.4f}"
    )
    print(
        f"- Confusion (TN/FP/FN/TP): {int(metrics_mah['tn'])}/{int(metrics_mah['fp'])}/"
        f"{int(metrics_mah['fn'])}/{int(metrics_mah['tp'])}"
    )

    print("\n[산출물]")
    print(f"- {out_dir / 'embedding_tsne_umap.png'}")
    print(f"- {out_dir / 'roc_pr_mahalanobis.png'}")
    print(f"- {out_dir / 'score_hist_mahalanobis.png'}")


if __name__ == "__main__":
    main()
