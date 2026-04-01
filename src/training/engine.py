import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


@dataclass
class TrainHistory:
    train_loss: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_acc: List[float]


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_acc += accuracy_from_logits(logits, targets) * batch_size

    return total_loss / total_samples, total_acc / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_acc += accuracy_from_logits(logits, targets) * batch_size

    return total_loss / total_samples, total_acc / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Tuple[TrainHistory, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = TrainHistory(train_loss=[], train_acc=[], val_loss=[], val_acc=[])
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        history.train_loss.append(tr_loss)
        history.train_acc.append(tr_acc)
        history.val_loss.append(va_loss)
        history.val_acc.append(va_acc)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
            f"Val Loss: {va_loss:.4f}, Val Acc: {va_acc:.4f}"
        )

    elapsed = time.perf_counter() - start
    return history, elapsed


def print_history_summary(title: str, history: TrainHistory) -> None:
    print(f"\n[{title}] 학습 요약")
    for i in range(len(history.train_loss)):
        print(
            f"  Epoch {i + 1:02d}: "
            f"train_loss={history.train_loss[i]:.4f}, train_acc={history.train_acc[i]:.4f}, "
            f"val_loss={history.val_loss[i]:.4f}, val_acc={history.val_acc[i]:.4f}"
        )


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    print("\n================ 최종 비교 (MNIST) ================")
    print(
        "모델".ljust(36)
        + "Test Loss".rjust(12)
        + "Test Acc".rjust(12)
        + "Time(s)".rjust(12)
        + "Trainable Params".rjust(20)
    )
    print("-" * 92)

    for model_name, vals in results.items():
        print(
            model_name.ljust(36)
            + f"{vals['test_loss']:.4f}".rjust(12)
            + f"{vals['test_acc']:.4f}".rjust(12)
            + f"{vals['train_time_sec']:.2f}".rjust(12)
            + f"{int(vals['trainable_params'])}".rjust(20)
        )

    print("=" * 92)
