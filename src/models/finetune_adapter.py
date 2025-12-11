import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from src.utils import calculate_metrics


class MLP(nn.Module):
    """A simple multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class FineTuneAdapter(pl.LightningModule):
    """PyTorch Lightning module to fine-tune a model with an MLP adapter."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.mlp = MLP(input_dim, hidden_dim)
        self.lr = lr

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the adapter."""
        return self.mlp(x)

    def _shared_step(self, batch: tuple[Tensor, Tensor], stage: str) -> Tensor:
        """
        A shared step for training, validation, and testing to reduce code duplication.
        """
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = y.detach().cpu().numpy()

        metrics = calculate_metrics(probs, targets, threshold=0.5)

        log_values = {f"{stage}/{metric}": value for metric, value in metrics.items()}
        log_values[f"{stage}/loss"] = loss.detach()

        self.log_dict(log_values)
        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def compute_loss(self, logits: Tensor, y: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(logits, y.float())
