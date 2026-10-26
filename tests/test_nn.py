import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from bank_churn.nn import MLP, TorchMLPClassifier, train_step


def toy_problem(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def test_forward_pass_returns_one_logit_per_row():
    model = MLP(n_features=13, hidden_sizes=(32, 16), dropout=0.1)
    assert model(torch.zeros(7, 13)).shape == (7,)


def test_one_training_step_lowers_the_loss():
    torch.manual_seed(0)
    X, y = toy_problem(64)
    x, t = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    model = MLP(n_features=5, hidden_sizes=(16,), dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.BCEWithLogitsLoss()
    before = train_step(model, optimizer, loss_fn, x, t)
    with torch.no_grad():
        after = loss_fn(model(x), t).item()
    assert after < before


def test_classifier_learns_a_separable_problem():
    X, y = toy_problem()
    clf = TorchMLPClassifier(max_epochs=60, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert roc_auc_score(y, proba[:, 1]) > 0.95
    assert set(clf.predict(X)) <= {0, 1}
    assert 1 <= clf.n_epochs_ <= 60


def test_classifier_is_reproducible_with_a_seed():
    X, y = toy_problem()
    first = TorchMLPClassifier(max_epochs=5, random_state=3).fit(X, y).predict_proba(X)
    second = TorchMLPClassifier(max_epochs=5, random_state=3).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(first, second)


def test_classifier_needs_both_classes():
    X, _ = toy_problem(50)
    with pytest.raises(ValueError, match="both classes"):
        TorchMLPClassifier().fit(X, np.zeros(50, dtype=int))
