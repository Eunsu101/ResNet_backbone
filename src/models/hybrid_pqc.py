import math

import pennylane as qml
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNetFeatureExtractor(nn.Module):
    """ResNet18의 fc 이전까지를 feature extractor로 사용한다."""

    def __init__(self):
        super().__init__()
        try:
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception as exc:
            print(f"[경고] pretrained 로드 실패: {exc}")
            print("[경고] weights=None으로 fallback 합니다.")
            backbone = resnet18(weights=None)

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.out_features = backbone.fc.in_features

        # 실험 조건: feature extractor는 고정
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.features(x)
            feat = torch.flatten(feat, 1)
        return feat


class PQCHead(nn.Module):
    """
    고전 feature를 PQC로 분류하는 헤드.

    feature -> projection -> PQC -> linear classifier
    """

    def __init__(self, in_features: int, num_classes: int, n_qubits: int, q_layers: int):
        super().__init__()
        self.projection = nn.Linear(in_features, n_qubits)

        # 노이즈 미고려 조건에 맞는 시뮬레이터
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (q_layers, n_qubits)}
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 임베딩 안정성을 위해 입력 범위를 제한
        projected = torch.tanh(self.projection(features)) * math.pi
        quantum_out = self.quantum_layer(projected)
        return self.classifier(quantum_out)
