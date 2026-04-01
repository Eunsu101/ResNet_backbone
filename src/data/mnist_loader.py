from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BinaryAnomalyWrapper(Dataset):
    """라벨을 정상=0 / 비정상=1로 변환하는 데이터셋 래퍼."""

    def __init__(self, base_dataset: Dataset, normal_digit: int = 0):
        self.base_dataset = base_dataset
        self.normal_digit = normal_digit

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]
        binary_label = 0 if int(label) == self.normal_digit else 1
        return image, binary_label


def build_mnist_dataloaders(
    data_dir: str,
    batch_size: int,
    val_size: int,
    image_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    MNIST를 ResNet 입력 형태(3채널, resize)로 변환한 dataloader를 만든다.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            # Windows 멀티프로세싱 호환을 위해 lambda 대신 표준 transform을 사용한다.
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    if val_size <= 0 or val_size >= len(train_full):
        raise ValueError("val_size는 1 이상, train 전체 미만이어야 합니다.")

    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size])

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def build_mnist_oneclass_dataloaders(
    data_dir: str,
    batch_size: int,
    val_size: int,
    image_size: int,
    num_workers: int,
    seed: int,
    normal_digit: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    One-class 평가용 dataloader를 만든다.

    - train: 정상(normal_digit) 샘플만 포함
    - val/test: 정상=0, 비정상=1 이진 라벨
    """
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    if val_size <= 0 or val_size >= len(train_full):
        raise ValueError("val_size는 1 이상, train 전체 미만이어야 합니다.")

    generator = torch.Generator().manual_seed(seed)
    train_size = len(train_full) - val_size
    train_candidate, val_set = random_split(train_full, [train_size, val_size], generator=generator)

    full_targets = train_full.targets
    train_candidate_indices = torch.tensor(train_candidate.indices, dtype=torch.long)
    normal_mask = full_targets[train_candidate_indices] == normal_digit
    normal_indices = train_candidate_indices[normal_mask].tolist()

    if len(normal_indices) == 0:
        raise RuntimeError("train split에서 정상 샘플을 찾지 못했습니다.")

    train_normal_set = Subset(train_full, normal_indices)
    val_binary = BinaryAnomalyWrapper(val_set, normal_digit=normal_digit)
    test_binary = BinaryAnomalyWrapper(test_set, normal_digit=normal_digit)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_normal_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_binary,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_binary,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
