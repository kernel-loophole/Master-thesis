from typing import Dict, Any, Tuple, List, Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    nn = type("nn", (), {"Module": object})
    torch = None


class LinearProbe(nn.Module):
    """Linear probe classifier on activation representations: y = W * h + b."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        if torch is not None:
            self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: Any) -> Any:
        if torch is None:
            return x
        return self.linear(x)

    def get_direction(self, class_idx: int) -> Any:
        """Extract learned linear direction weight vector for target concept."""
        if torch is None:
            return None
        return self.linear.weight[class_idx].detach()


class LinearProbeTrainer:
    """Trains a linear probe model on activations to predict spatial concepts."""

    def __init__(self, probe: LinearProbe, device: str = "cpu"):
        self.probe = probe
        self.device = device
        if torch is not None:
            self.probe.to(device)

    def train(
        self,
        X_train: Any,
        y_train: Any,
        epochs: int = 50,
        lr: float = 1e-3
    ) -> Dict[str, float]:
        if torch is None:
            return {"train_loss": 0.0, "train_accuracy": 0.0}

        optimizer = torch.optim.Adam(self.probe.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device) if not isinstance(X_train, torch.Tensor) else X_train.to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(self.device) if not isinstance(y_train, torch.Tensor) else y_train.to(self.device)

        self.probe.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.probe(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        self.probe.eval()
        with torch.no_grad():
            preds = torch.argmax(self.probe(X_train), dim=-1)
            acc = (preds == y_train).float().mean().item()

        return {"train_loss": loss.item(), "train_accuracy": acc}


class LinearIntervention:
    """Direct linear probe intervention baseline: h' = h + alpha * w_probe."""

    def __init__(self, probe_direction: Any, alpha: float = 1.0):
        self.direction = probe_direction
        self.alpha = alpha

    def __call__(self, h: Any) -> Any:
        if torch is None or not isinstance(h, torch.Tensor):
            return h
        norm_direction = self.direction / (torch.norm(self.direction) + 1e-8)
        return h + self.alpha * norm_direction.to(h.device)
