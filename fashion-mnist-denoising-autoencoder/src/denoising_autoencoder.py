"""
Denoising autoencoder for Fashion-MNIST.

Trains a fully connected autoencoder to recover clean 28x28 clothing images
from inputs corrupted with heavy additive Gaussian noise (sigma = 100/255),
then reports reconstruction MSE / PSNR and saves qualitative result plots.

Usage:
    python src/denoising_autoencoder.py                 # defaults used for the reported results
    python src/denoising_autoencoder.py --hidden 512 256 512 --epochs 30
"""

import argparse
import json
import os
import time

import cv2
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

SEED = 883324          # fixed seed used for the reported results
NOISE_STD = 100.0      # Gaussian noise std in raw pixel units (0-255)
TRAIN_FRACTION = 0.8   # fraction of the training set sampled for training


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def make_reproducible(seed: int = SEED) -> None:
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data(noise_std: float = NOISE_STD):
    """Return (x_train, y_train, x_test, y_test): x = noisy input, y = clean target, scaled to [0, 1]."""
    (train_imgs, _), (test_imgs, _) = keras.datasets.fashion_mnist.load_data()

    idx = np.random.randint(train_imgs.shape[0], size=int(train_imgs.shape[0] * TRAIN_FRACTION))
    y_train = train_imgs[idx]
    x_train = y_train + np.random.normal(0, noise_std, y_train.shape)

    y_test = test_imgs
    x_test = y_test + np.random.normal(0, noise_std, y_test.shape)

    scale = lambda a: a.astype("float32") / 255.0
    return scale(x_train), scale(y_train), scale(x_test), scale(y_test)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(x_train, hidden_units, learning_rate, l2=1e-5) -> keras.Model:
    """Flatten -> per-pixel standardisation -> dense ReLU hidden layers -> linear 784 output -> reshape."""
    flatten = keras.layers.Flatten()
    norm = keras.layers.Normalization()
    norm.adapt(flatten(x_train))  # learn 784 per-pixel means/variances before training

    inputs = keras.Input(shape=(28, 28))
    x = norm(flatten(inputs))
    for units in hidden_units:
        x = keras.layers.Dense(units, activation="relu", kernel_regularizer=keras.regularizers.L2(l2))(x)
    x = keras.layers.Dense(784, activation="linear", kernel_regularizer=keras.regularizers.L2(l2))(x)
    outputs = keras.layers.Reshape((28, 28))(x)

    model = keras.Model(inputs, outputs, name="denoising_autoencoder")
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer=optimizer)
    return model


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def mse(a, b) -> float:
    return float(np.mean((a - b) ** 2))


def psnr(mse_value: float, peak: float = 1.0) -> float:
    return float(10 * np.log10(peak ** 2 / mse_value))


def plot_triplets(noisy, clean, pred, title, path, n=5):
    fig, axes = plt.subplots(nrows=3, ncols=n, figsize=(11, 2.4))
    fig.suptitle(title)
    for i in range(n):
        for row, img in enumerate((noisy[i], clean[i], pred[i])):
            axes[row, i].imshow(img, cmap="gray")
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])
    for row, label in enumerate(("Noisy", "Clean", "Output")):
        axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_generation_from_noise(model, path, n=10):
    """Feed smooth random blobs + strong noise to probe what the decoder has learned to 'imagine'."""
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(11, 2.4))
    fig.suptitle("Outputs from pure-noise inputs")
    for i in range(n):
        img = cv2.resize(np.random.normal(0, 10, (2, 4)), dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        img = (img + np.random.normal(0, 300, (28, 28))).reshape(1, 28, 28) / 255.0
        out = model.predict(img, verbose=0)
        axes[0, i].imshow(img[0], cmap="gray")
        axes[1, i].imshow(out[0], cmap="gray")
        for r in range(2):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int, nargs="+", default=[300], help="hidden layer sizes, e.g. 512 128 512")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    make_reproducible()
    x_train, y_train, x_test, y_test = load_data()
    print(f"train: {x_train.shape}  test: {x_test.shape}")

    make_reproducible()
    model = build_model(x_train, args.hidden, args.lr)
    model.summary()

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-4, verbose=1)
    make_reproducible()
    t0 = time.time()
    history = model.fit(x_train, y_train, batch_size=args.batch, epochs=args.epochs,
                        shuffle=True, verbose=1, callbacks=[reduce_lr])
    elapsed = time.time() - t0

    pred_train = model.predict(x_train, verbose=0)
    pred_test = model.predict(x_test, verbose=0)

    metrics = {
        "hidden_units": args.hidden,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "train_time_s": round(elapsed, 1),
        "noisy_input_mse_test": mse(x_test, y_test),
        "train_mse": mse(pred_train, y_train),
        "test_mse": mse(pred_test, y_test),
    }
    metrics["noisy_input_psnr_db"] = psnr(metrics["noisy_input_mse_test"])
    metrics["test_psnr_db"] = psnr(metrics["test_mse"])
    metrics["final_epoch_loss"] = float(history.history["loss"][-1])

    print(json.dumps(metrics, indent=2))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_triplets(x_train, y_train, pred_train, "Denoising results (train)", os.path.join(args.out, "denoising_train.png"))
    plot_triplets(x_test, y_test, pred_test, "Denoising results (test)", os.path.join(args.out, "denoising_test.png"))
    plot_generation_from_noise(model, os.path.join(args.out, "generation_from_noise.png"))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(history.history["loss"]) + 1), history.history["loss"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(args.out, "training_loss.png"), bbox_inches="tight", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
