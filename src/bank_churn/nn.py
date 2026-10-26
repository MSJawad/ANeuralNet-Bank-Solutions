"""A small PyTorch feed-forward network wrapped as a scikit-learn classifier."""

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from torch import nn


class MLP(nn.Module):
    """Fully connected network that outputs one churn logit per row."""

    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for size in hidden_sizes:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Run one optimisation step on a batch and return the batch loss before the update."""
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.item()


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Binary classifier trained with Adam on binary cross-entropy, with early stopping.

    A stratified ``validation_fraction`` of the training rows is set aside to decide when to
    stop: training ends after ``patience`` epochs without a lower validation loss, and the
    weights from the best epoch are kept. Expects already-scaled numeric input.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (32, 16),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 15,
        validation_fraction: float = 0.15,
        random_state: int = 0,
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("TorchMLPClassifier needs both classes 0 and 1 in y")
        y = y.astype(np.float32)

        torch.manual_seed(self.random_state)
        generator = torch.Generator().manual_seed(self.random_state)

        X_fit, X_stop, y_fit, y_stop = train_test_split(
            X,
            y,
            test_size=self.validation_fraction,
            stratify=y,
            random_state=self.random_state,
        )
        x_fit, t_fit = torch.from_numpy(X_fit), torch.from_numpy(y_fit)
        x_stop, t_stop = torch.from_numpy(X_stop), torch.from_numpy(y_stop)

        model = MLP(X.shape[1], tuple(self.hidden_sizes), self.dropout)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.history_ = []
        for _ in range(self.max_epochs):
            order = torch.randperm(len(x_fit), generator=generator)
            train_losses = []
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                train_losses.append(
                    train_step(model, optimizer, loss_fn, x_fit[batch], t_fit[batch])
                )
            model.eval()
            with torch.no_grad():
                stop_loss = loss_fn(model(x_stop), t_stop).item()
            self.history_.append(
                {"train_loss": float(np.mean(train_losses)), "val_loss": stop_loss}
            )
            if stop_loss < best_loss:
                best_loss = stop_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.n_features_in_ = X.shape[1]
        self.n_epochs_ = len(self.history_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        x = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            positive = torch.sigmoid(self.model_(x)).numpy().astype(np.float64)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
