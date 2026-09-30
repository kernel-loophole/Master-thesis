import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from spatial_sae.sae.model import SparseAutoencoder
from spatial_sae.sae.loss import SAELoss
from spatial_sae.sae.metrics import compute_sae_metrics
from spatial_sae.utils.logging import get_logger
from spatial_sae.utils.checkpoints import save_checkpoint

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable, **kwargs: iterable

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None

logger = get_logger("spatial_sae.trainer")


class SAETrainer:
    """Trainer class for training Sparse Autoencoders on extracted activations."""

    def __init__(
        self,
        sae: SparseAutoencoder,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        sparsity_coefficient: float = 0.001,
        output_dir: Union[str, Path] = "./outputs/sae",
        device: str = "cuda",
        use_amp: bool = True
    ):
        self.sae = sae
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.use_amp = use_amp

        if torch is not None and hasattr(self.sae, "to"):
            try:
                self.sae.to(device)
                self.optimizer = torch.optim.AdamW(
                    self.sae.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
                self.scaler = torch.amp.GradScaler('cuda') if (use_amp and device == "cuda") else None
            except Exception:
                self.optimizer = None
                self.scaler = None
        else:
            self.optimizer = None
            self.scaler = None

        self.loss_fn = SAELoss(sparsity_coefficient=sparsity_coefficient)
        self.feature_activation_counts = None

    def train(self, epochs: int = 25, checkpoint_freq: int = 5) -> Dict[str, Any]:
        """Run training loop over specified number of epochs."""
        logger.info(f"Starting SAE training for {epochs} epochs on device: {self.device}")

        history = {"train_loss": [], "val_loss": [], "explained_variance": [], "l0_norm": []}

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._eval_epoch() if self.val_loader else {}

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
                f"Train Loss: {train_metrics['total_loss']:.4f} (Rec: {train_metrics['rec_loss']:.4f}, L1: {train_metrics['l1_loss']:.4f}) | "
                f"ExpVar: {train_metrics.get('explained_variance', 0.0):.4f} | L0: {train_metrics.get('l0_norm', 0.0):.1f}"
            )

            history["train_loss"].append(train_metrics["total_loss"])
            history["explained_variance"].append(train_metrics.get("explained_variance", 0.0))
            history["l0_norm"].append(train_metrics.get("l0_norm", 0.0))

            if val_metrics:
                history["val_loss"].append(val_metrics.get("total_loss", 0.0))

            # Periodic checkpoint saving
            if epoch % checkpoint_freq == 0 or epoch == epochs:
                ckpt_path = self.output_dir / f"sae_epoch_{epoch:03d}.pt"
                meta = {
                    "epoch": epoch,
                    "metrics": train_metrics,
                    "latent_dim": self.sae.latent_dim,
                    "input_dim": self.sae.input_dim
                }
                state_dict = self.sae.state_dict() if hasattr(self.sae, "state_dict") else {}
                save_checkpoint(state_dict, meta, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

        return history

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        if torch is None or self.train_loader is None or self.optimizer is None:
            return {"total_loss": 0.0, "rec_loss": 0.0, "l1_loss": 0.0}

        self.sae.train()
        total_loss_accum = 0.0
        rec_loss_accum = 0.0
        l1_loss_accum = 0.0
        exp_var_accum = 0.0
        l0_accum = 0.0
        steps = 0

        for batch in self.train_loader:
            x = batch["activations"] if isinstance(batch, dict) else batch
            x = x.to(self.device).float()

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    x_hat, z = self.sae(x)
                    loss, metrics = self.loss_fn(x, x_hat, z)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                x_hat, z = self.sae(x)
                loss, metrics = self.loss_fn(x, x_hat, z)
                loss.backward()
                self.optimizer.step()

            # Enforce unit norm decoder constraint
            if hasattr(self.sae, "normalize_decoder_weights"):
                self.sae.normalize_decoder_weights()

            total_loss_accum += metrics["total_loss"]
            rec_loss_accum += metrics["rec_loss"]
            l1_loss_accum += metrics["l1_loss"]
            exp_var_accum += metrics["explained_variance"]
            l0_accum += metrics["l0_norm"]
            steps += 1

        return {
            "total_loss": total_loss_accum / max(1, steps),
            "rec_loss": rec_loss_accum / max(1, steps),
            "l1_loss": l1_loss_accum / max(1, steps),
            "explained_variance": exp_var_accum / max(1, steps),
            "l0_norm": l0_accum / max(1, steps),
        }

    def _eval_epoch(self) -> Dict[str, float]:
        if torch is None or self.val_loader is None:
            return {}

        self.sae.eval()
        total_loss_accum = 0.0
        steps = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["activations"] if isinstance(batch, dict) else batch
                x = x.to(self.device).float()
                x_hat, z = self.sae(x)
                loss, metrics = self.loss_fn(x, x_hat, z)
                total_loss_accum += metrics["total_loss"]
                steps += 1

        return {"total_loss": total_loss_accum / max(1, steps)}
