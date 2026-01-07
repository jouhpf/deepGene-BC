import os
from typing import Optional, Dict, Any, Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """
    early stopping strategy to halt training when validation loss stops improving.
    Args:
        patience (int):
            Number of epochs with no improvement after which training will be stopped.
        delta (float):
            Minimum change in the monitored quantity to qualify as an improvement.
    """

    def __init__(self, patience: int = 5, delta: float = 0.001) -> None:
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss <= self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # reset the best loss and counter
            self.best_loss = val_loss
            self.counter = 0


def accuracy_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if output.shape[1] == 1:
        predicts = (output > 0.5).int()
        return (predicts == target).float().mean()
    else:
        _, predicts = torch.max(output, 1)
        return (predicts == target).float().mean()


class Trainer:
    """
    a unified training interface for PyTorch trained_models.

    this class encapsulates the training process, including the training loop,
    validation, checkpoint saving/loading, early stop

    Args:
        model (nn.Module):
            The neural network model to train.
        device (torch.device):
            The device to use for computation.
        train_loader (DataLoader):
            DataLoader for the training set.
        val_loader (DataLoader):
            DataLoader for the validation set.
        loss_fn:
            Loss function.
        optimizer (optim.Optimizer):
            Optimizer.
        lr_scheduler_fn:
            Learning rate scheduler.
        checkpoint_dir (str): Directory to save checkpoints.
    """

    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: optim.Optimizer,
            loss_fn,
            test_loader=None,
            lr_scheduler_fn=None,
            checkpoint_dir: str = './checkpoints',
            metrics: Optional[Dict[str, Callable]] = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler_fn = lr_scheduler_fn
        self.checkpoint_dir = checkpoint_dir
        self.metrics = metrics if metrics else {}
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _train_epoch(self, epoch: int, total_epoch: int):
        """
        Train the model for a single epoch.

        Args:
            epoch (int): Current epoch number.
        Returns:
            float: Average training loss for the epoch.
        """
        # init train model
        self.model.train()

        running_loss = 0.0
        total_samples = 0
        metric_sums = {name: 0.0 for name in self.metrics}

        progress_bar = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc=f"Epoch {epoch}/{total_epoch} [Train]"
        )

        for data, target in progress_bar:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

            batch_size = data.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            # Calculate metrics
            for name, metric_fn in self.metrics.items():
                metric_sums[name] += metric_fn(output, target).item() * batch_size

            progress_bar.set_postfix({
                'train_loss': f"{running_loss / total_samples:.4f}",
                **{name: f"{metric_sums[name] / total_samples:.4f}" for name in self.metrics}
            })

        epoch_loss = running_loss / total_samples
        epoch_metrics = {name: metric_sums[name] / total_samples for name in self.metrics}
        return epoch_loss, epoch_metrics

    @torch.no_grad()
    def _evaluate_epoch(self, loader, epoch: int, total_epoch: int):
        """
        Validate the model for a single epoch.

        Args:
            epoch (int): Current epoch number.
        Returns:
            float: Average validation loss for the epoch.
        """
        # init evaluation mode
        self.model.eval()

        val_loss = 0.0
        total_samples = 0
        metric_sums = {name: 0.0 for name in self.metrics}

        progress_bar = tqdm(
            loader,
            total=len(loader),
            desc=f"Epoch {epoch}/{total_epoch} [Val]"
        )

        for data, target in progress_bar:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.loss_fn(output, target)
            batch_size = data.size(0)
            val_loss += loss.item() * batch_size
            total_samples += batch_size
            # Calculate metric
            for name, metric_fn in self.metrics.items():
                metric_sums[name] += metric_fn(output, target).item() * batch_size

            progress_bar.set_postfix({
                'val_loss': f"{val_loss / total_samples:.4f}",
                **{name: f"{metric_sums[name] / total_samples:.4f}" for name in self.metrics}
            })
        epoch_loss = val_loss / total_samples
        epoch_metrics = {name: metric_sums[name] / total_samples for name in self.metrics}
        return epoch_loss, epoch_metrics

    def fit(
            self,
            epochs: int,
            start_epoch: int = 1,
            save_best: bool = True,
            save_every_epoch: bool = True,
            early_stop: Optional[EarlyStopping] = None,
            monitor: str = 'val_loss'
    ) -> tuple[dict[str, list[Any]], int]:
        """
        train the model for multiple epochs with validation and checkpointing.

        Args:
            epochs (int):
                Total number of training epochs.
            start_epoch (int):
                Starting epoch number (default is 1).
            save_best (bool):
                Whether to save the best model based on validation loss.
            save_every_epoch (bool):
                save check of every epoch or not
            early_stop (Optional[EarlyStopping]):
                Early stopping callback.
            monitor (str): Metric to monitor the for saving best model (default: 'val_loss').
        Returns:
            Dict[str, Any]: A history dictionary containing train_loss and val_loss.
        """
        print(f"training on {self.device}")

        best_epoch = 0
        best_monitor_value = float('inf') if 'loss' in monitor else 0.0

        history = {'train_loss': [], 'val_loss': []}
        for name in self.metrics:
            history[f'train_{name}'] = []
            history[f'val_{name}'] = []

        for epoch in range(start_epoch, epochs + 1):
            train_loss, train_metrics = self._train_epoch(epoch, epochs)
            history['train_loss'].append(train_loss)
            for name, value in train_metrics.items():
                history[f'train_{name}'].append(value)

            val_loss, val_metrics = self._evaluate_epoch(self.val_loader, epoch, epochs)
            history['val_loss'].append(val_loss)
            for name, value in val_metrics.items():
                history[f'val_{name}'].append(value)

            current_monitor = val_loss if 'loss' in monitor else val_metrics.get(monitor, None)

            if save_best and current_monitor is not None:
                if ('loss' in monitor and current_monitor < best_monitor_value) or \
                        ('loss' not in monitor and current_monitor > best_monitor_value):
                    best_monitor_value = current_monitor
                    best_epoch = epoch
                    self.save_checkpoint(epoch, best=True)
                    print(f"Saved best model at epoch {epoch} ({monitor}: {best_monitor_value:.4f})")

            if self.lr_scheduler_fn is not None:
                self.lr_scheduler_fn.step()

            # save check of every epoch
            if save_every_epoch:
                self.save_checkpoint(epoch, best=False)

            if early_stop is not None:
                early_stop(val_loss)
                if early_stop.early_stop:
                    print("Early stopping triggered")
                    return history, best_epoch
        return history, best_epoch

    def test(self):
        return self._evaluate_epoch(self.test_loader, 0, 0)

    def save_checkpoint(self, epoch: int, best: bool = False) -> None:
        """
        save model checkpoint to disk.

        Args:
            epoch (int): Current epoch number.
            best (bool): Whether this is the best model.
        """
        state: Dict[str, Any] = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        filename: str = f"{'best_' if best else ''}checkpoint_epoch_{epoch}.pth"
        path: str = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> Optional[int]:
        """
        load model checkpoint from disk.

        Args:
            path (str): Path to the checkpoint file.
            load_optimizer (bool): Whether to load the optimizer state.
        Returns:
            Optional[int]: Epoch number to resume training from, or None if unavailable.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        return checkpoint.get('epoch', None)
