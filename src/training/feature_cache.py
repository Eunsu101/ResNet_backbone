import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def build_feature_cache(
    extractor: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> TensorDataset:
    """
    고정된 ResNet feature를 선계산해서 TensorDataset으로 캐시한다.

    이 단계로 PQC 학습 병목(매 스텝 이미지 feature 재추출)을 줄인다.
    """
    extractor.eval()
    features_all = []
    labels_all = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        feats = extractor(images)
        features_all.append(feats.cpu())
        labels_all.append(labels.cpu())

    return TensorDataset(torch.cat(features_all, dim=0), torch.cat(labels_all, dim=0))
