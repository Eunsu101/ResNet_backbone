from pathlib import Path
import importlib
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def compute_euclidean_scores(features: torch.Tensor, center: torch.Tensor) -> np.ndarray:
    feats = _to_numpy(features)
    c = _to_numpy(center)
    return np.linalg.norm(feats - c[None, :], axis=1)


def fit_gaussian_from_normal(features_normal: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = features_normal.float()
    mean = feats.mean(dim=0)
    centered = feats - mean
    cov = (centered.T @ centered) / max(feats.shape[0] - 1, 1)
    cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    cov_inv = torch.linalg.pinv(cov)
    return mean, cov_inv


def compute_mahalanobis_scores(features: torch.Tensor, mean: torch.Tensor, cov_inv: torch.Tensor) -> np.ndarray:
    x = features.float() - mean.unsqueeze(0)
    left = x @ cov_inv
    sq_dist = (left * x).sum(dim=1)
    return torch.sqrt(torch.clamp(sq_dist, min=0.0)).detach().cpu().numpy()


def threshold_from_normal(scores: np.ndarray, quantile: float = 0.95) -> float:
    return float(np.quantile(scores, quantile))


def compute_binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (scores >= threshold).astype(np.int64)

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)

    fpr95 = np.nan
    tpr_at_fpr05 = np.nan

    mask_tpr95 = tpr >= 0.95
    if np.any(mask_tpr95):
        fpr95 = float(np.min(fpr[mask_tpr95]))

    mask_fpr05 = fpr <= 0.05
    if np.any(mask_fpr05):
        tpr_at_fpr05 = float(np.max(tpr[mask_fpr05]))

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "fpr95": float(fpr95),
        "tpr_at_fpr05": float(tpr_at_fpr05),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def plot_roc_pr(labels: np.ndarray, scores: np.ndarray, out_path: Path, title_prefix: str) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, label="ROC")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title(f"{title_prefix} ROC")
    axes[0].grid(alpha=0.3)

    axes[1].plot(recall, precision, label="PR")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{title_prefix} Precision-Recall")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_score_histogram(labels: np.ndarray, scores: np.ndarray, threshold: float, out_path: Path, title: str) -> None:
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(normal_scores, bins=50, alpha=0.6, label="normal (0)")
    ax.hist(anomaly_scores, bins=50, alpha=0.6, label="anomaly (1-9)")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"threshold={threshold:.4f}")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_tsne_umap(features: np.ndarray, labels: np.ndarray, out_path: Path, seed: int, title: str) -> None:
    pca = PCA(n_components=min(50, features.shape[1], features.shape[0] - 1), random_state=seed)
    reduced = pca.fit_transform(features)

    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=seed)
    emb_tsne = tsne.fit_transform(reduced)

    fig_cols = 2
    fig, axes = plt.subplots(1, fig_cols, figsize=(12, 5))

    axes[0].scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=labels, s=6, alpha=0.8, cmap="coolwarm")
    axes[0].set_title("t-SNE (PCA-50 pre)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    try:
        umap_module = importlib.import_module("umap")
        umap_model = umap_module.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=seed)
        emb_umap = umap_model.fit_transform(reduced)
        axes[1].scatter(emb_umap[:, 0], emb_umap[:, 1], c=labels, s=6, alpha=0.8, cmap="coolwarm")
        axes[1].set_title("UMAP (PCA-50 pre)")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    except Exception:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "UMAP unavailable\nInstall umap-learn", ha="center", va="center")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
