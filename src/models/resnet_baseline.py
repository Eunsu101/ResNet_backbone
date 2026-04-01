import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18_baseline(num_classes: int = 10) -> nn.Module:
    """ImageNet pretrained ResNet18 baseline 모델을 생성한다."""
    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception as exc:
        print(f"[경고] pretrained 로드 실패: {exc}")
        print("[경고] weights=None으로 fallback 합니다.")
        model = resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
