import random

import torch


def seed_everything(seed: int) -> None:
    """실험 재현성을 위해 랜덤 시드를 고정한다."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """CUDA 사용 가능 시 GPU를 선택한다."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
