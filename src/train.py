"""Build a model and train it. YOU write the bodies of these functions.

The model is ResNet18, a convolutional network that was already trained on
ImageNet, 1.2 million photographs of 1,000 kinds of thing. You keep everything
it learned there and replace only its last layer, so that it answers with your
categories instead of those 1,000. Then you train it further on your images.

Run `pytest tests/test_train.py` after you fill them in. The first run
downloads the ImageNet weights, about 45 MB.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def pick_device():
    """A CUDA GPU if there is one, then an Apple GPU, then the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes, pretrained=True, freeze=False):
    """Return a ResNet18 whose last layer has `num_classes` outputs.

    Arguments
        num_classes  int, how many categories you have
        pretrained   True: start from the ImageNet weights,
                     torchvision.models.ResNet18_Weights.IMAGENET1K_V1.
                     False: start from random numbers, as if ImageNet had
                     never happened. You need this for the comparison in
                     Problem 3.
        freeze       True: only the new last layer is trained. Every other
                     parameter keeps the value it came with, which you do by
                     setting requires_grad to False on it.
                     False: every parameter is trained.

    Returns the model from torchvision.models.resnet18, with its `fc`
    attribute replaced by a new nn.Linear that has 512 inputs and num_classes
    outputs.

    Do not put a softmax at the end. The loss function adds it for you, and the
    web page adds it for you.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # The new layer is made after freezing, so it is the only trainable part.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(model, X_train, y_train, X_test, y_test,
          epochs=10, lr=1e-4, batch_size=32):
    """Train the model and report what happened at every epoch.

    Arguments
        model        what build_model returned, or any other torch model
        X_train      np.float32 (N, ...)      y_train  np.int64 (N,)
        X_test       np.float32 (M, ...)      y_test   np.int64 (M,)
        epochs       int, how many times to go through the training set
        lr           float, the learning rate
        batch_size   int, how many rows per step

    Returns a dict named history with four keys. Each value is a list of
    `epochs` numbers, one per epoch:
        "train_loss"  the mean loss over the batches of that epoch
        "train_acc"   the share of training rows the model got right in those
                      batches, between 0.0 and 1.0
        "test_loss"   the mean loss over the test set, measured after the epoch
        "test_acc"    the share of the test set it gets right, measured after
                      the epoch, between 0.0 and 1.0

    What to do
        Use a GPU when there is one:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Move the model there once, and each batch there as you use it. Keep
        X_train and X_test themselves in ordinary memory: they may not fit on
        the GPU.
        Shuffle the training rows again at the start of every epoch.
        Use nn.CrossEntropyLoss, which expects raw outputs, and torch.optim.Adam
        over the parameters that have requires_grad set.
        Call model.train() before the training batches and model.eval() before
        measuring the test set. ResNet contains BatchNorm layers, which behave
        differently in the two modes.
        Do not compute gradients while measuring the test set, and never let
        the test rows change the parameters.
        Print one line per epoch, so that you can watch it while it runs.

    The model is trained in place. When this function returns, `model` is the
    trained one, and it is left on the device it was trained on.
    """
    device = pick_device()
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    rng = np.random.default_rng(0)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(X_train))
        batch_losses, correct = [], 0
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            xb = torch.from_numpy(X_train[rows]).to(device)
            yb = torch.from_numpy(y_train[rows]).to(device)

            optimiser.zero_grad()
            outputs = model(xb)
            loss = loss_fn(outputs, yb)
            loss.backward()
            optimiser.step()

            batch_losses.append(loss.item())
            correct += (outputs.argmax(dim=1) == yb).sum().item()

        test_loss, test_acc = _measure(model, X_test, y_test, loss_fn, device, batch_size)
        history["train_loss"].append(float(np.mean(batch_losses)))
        history["train_acc"].append(correct / len(X_train))
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"epoch {epoch:3d}/{epochs}  "
              f"train loss {history['train_loss'][-1]:.4f} acc {history['train_acc'][-1]:.1%}  "
              f"test loss {test_loss:.4f} acc {test_acc:.1%}", flush=True)

    return history


def _measure(model, X, y, loss_fn, device, batch_size):
    """Mean loss and accuracy on (X, y), without changing the model."""
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).to(device)
            yb = torch.from_numpy(y[start:start + batch_size]).to(device)
            outputs = model(xb)
            total_loss += loss_fn(outputs, yb).item() * len(yb)
            correct += (outputs.argmax(dim=1) == yb).sum().item()
    return total_loss / len(X), correct / len(X)


def count_trainable(model):
    """How many parameters training is allowed to change. Written for you."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save(model, path):
    """Save a trained model. Written for you."""
    torch.save(model, path)


def load(path):
    """Load a model saved by `save`, onto the CPU. Written for you."""
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model
