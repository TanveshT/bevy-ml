"""
PyTorch MNIST MLP — same architecture as ecs-ml for comparison.
784 → 256 → ReLU → 128 → ReLU → 10 → LogSoftmax
SGD lr=0.01, batch_size=32, 10 epochs

Reads MNIST IDX files directly from data/ (no torchvision download).
Outputs JSON metrics to stdout for comparison script.
"""

import json
import struct
import time
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import typer
from torch.utils.data import DataLoader, TensorDataset

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

app = typer.Typer()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def load_idx_images(path: str) -> np.ndarray:
    """Read IDX3 image file → numpy float32 array scaled to [0,1]."""
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Bad image magic: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def load_idx_labels(path: str) -> np.ndarray:
    """Read IDX1 label file → numpy int64 array."""
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Bad label magic: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def idx_path(data_dir: str, name: str) -> str:
    dot = os.path.join(data_dir, name.replace("-", "."))
    if os.path.isfile(dot):
        return dot
    dash = os.path.join(data_dir, name)
    if os.path.isfile(dash):
        return dash
    nested = os.path.join(data_dir, name, name)
    if os.path.isfile(nested):
        return nested
    raise FileNotFoundError(f"MNIST file not found: tried {dot}, {dash}, {nested}")


class MnistMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.log_softmax(self.fc3(x), dim=1)
        return x


def get_memory_mb() -> float:
    if HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return 0.0


@app.command()
def main(
    batch_size: int = typer.Option(32, help="Training batch size"),
    epochs: int = typer.Option(10, help="Number of training epochs"),
    lr: float = typer.Option(0.01, help="Learning rate"),
    seed: int = typer.Option(42, help="Random seed"),
    data_dir: str = typer.Option(
        os.path.join(PROJECT_ROOT, "data"), help="Path to MNIST IDX data directory"
    ),
):
    """Train an MLP on MNIST and output JSON metrics to stdout."""
    torch.manual_seed(seed)

    train_images = load_idx_images(idx_path(data_dir, "train-images-idx3-ubyte"))
    train_labels = load_idx_labels(idx_path(data_dir, "train-labels-idx1-ubyte"))
    test_images = load_idx_images(idx_path(data_dir, "t10k-images-idx3-ubyte"))
    test_labels = load_idx_labels(idx_path(data_dir, "t10k-labels-idx1-ubyte"))

    train_dataset = TensorDataset(torch.from_numpy(train_images), torch.from_numpy(train_labels))
    test_dataset = TensorDataset(torch.from_numpy(test_images), torch.from_numpy(test_labels))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MnistMLP()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    total_params = sum(p.numel() for p in model.parameters())

    results = {
        "framework": "pytorch",
        "architecture": "784->256->ReLU->128->ReLU->10->LogSoftmax",
        "optimizer": f"SGD(lr={lr})",
        "batch_size": batch_size,
        "total_params": total_params,
        "epochs": [],
    }

    mem_before = get_memory_mb()
    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        num_batches = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = output.argmax(dim=1)
            epoch_correct += (pred == target).sum().item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_acc = epoch_correct / (num_batches * batch_size) * 100.0

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += (pred == target).sum().item()
                test_total += target.size(0)

        test_acc = test_correct / test_total * 100.0
        epoch_time_ms = (time.time() - epoch_start) * 1000

        mem_current = get_memory_mb()

        epoch_result = {
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "train_acc": round(train_acc, 1),
            "test_acc": round(test_acc, 1),
            "time_ms": round(epoch_time_ms, 1),
            "memory_mb": round(mem_current, 1),
        }
        results["epochs"].append(epoch_result)

        print(
            f"Epoch {epoch}: loss={avg_loss:.4f} train_acc={train_acc:.1f}% "
            f"test_acc={test_acc:.1f}% time={epoch_time_ms:.0f}ms",
            file=sys.stderr,
        )

    total_time = (time.time() - total_start) * 1000
    results["total_time_ms"] = round(total_time, 1)
    results["peak_memory_mb"] = round(get_memory_mb(), 1)
    results["memory_before_mb"] = round(mem_before, 1)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
