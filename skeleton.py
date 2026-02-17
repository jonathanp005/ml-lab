#!/usr/bin/env python

from pathlib import Path
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from visualise import visualise


def hoeffding_unhappy_probability_bound(
    n_samples: int,
    empirical_error: float,
    tolerated_error: float,
) -> tuple[float, float]:
    """
    Hoeffding bounds for P(E_out > tolerated_error), given E_in.

    Returns
    -------
    one_sided, two_sided : tuple[float, float]
        one_sided: exp(-2 N eps^2)
        two_sided: 2 exp(-2 N eps^2)
    """
    epsilon = tolerated_error - empirical_error
    if epsilon <= 0:
        return 1.0, 1.0

    one_sided = float(np.exp(-2.0 * n_samples * epsilon ** 2))
    two_sided = float(min(1.0, 2.0 * one_sided))
    return one_sided, two_sided


def add_bias(X: np.ndarray) -> np.ndarray:
    """Append a bias term (1.0) to every sample."""
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def predict_from_augmented(X_aug: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Predict labels in {-1, +1} from already augmented data."""
    return np.where(X_aug @ w >= 0.0, 1, -1)


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Predict labels in {-1, +1} from non-augmented data."""
    return predict_from_augmented(add_bias(X), w)


def classification_accuracy(X: np.ndarray, Y: np.ndarray, w: np.ndarray) -> float:
    """Return classification accuracy for weights w."""
    return float(np.mean(predict(X, w) == Y))


def pla(
    data,
    plot_every: Optional[int] = 1,
    max_iteration: int = 100,
    random_update: bool = False,
    use_pocket: bool = False,
    random_seed: int = 0,
) -> tuple[np.ndarray, tuple[animation.ArtistAnimation, plt.Figure] | None, list[float]]:
    """
    Perceptron Learning Algorithm with optional pocket variant.

    Parameters
    ----------
    data : dict-like
        Data object with keys X and Y.
    plot_every : int | None
        Plot every k-th update. Disable with None/0.
    max_iteration : int
        Maximum number of perceptron updates.
    random_update : bool
        If True, update with a random misclassified sample.
    use_pocket : bool
        If True, return the best training-accuracy weights seen so far.
    random_seed : int
        Random seed used when random_update=True.
    """
    X = np.asarray(data["X"])
    Y = np.asarray(data["Y"]).reshape((-1,))
    X_aug = add_bias(X)

    w = np.zeros(X_aug.shape[1], dtype=float)
    best_w = w.copy()
    best_acc = classification_accuracy(X, Y, w)
    history: list[float] = []
    rng = np.random.default_rng(random_seed)

    fig, ax = None, None
    artists = []
    if plot_every:
        fig, ax = plt.subplots()

    for iteration in range(max_iteration):
        y_pred = predict_from_augmented(X_aug, w)
        misclassified = np.flatnonzero(y_pred != Y)
        acc = 1.0 - len(misclassified) / len(Y)
        history.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_w = w.copy()

        if plot_every and iteration % plot_every == 0:
            w_for_plot = best_w if use_pocket else w
            _fig, _ax, artists_ = visualise(
                w_for_plot,
                X,
                Y,
                ax=ax,
                animated=True,
                title=f"iter={iteration}, train_acc={acc:.3f}",
            )
            artists.append(artists_)

        if len(misclassified) == 0:
            break

        if random_update:
            idx = int(rng.choice(misclassified))
        else:
            idx = int(misclassified[0])
        w = w + Y[idx] * X_aug[idx]

    w_out = best_w if use_pocket else w
    if plot_every:
        ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=500, blit=False)
        return w_out, (ani, fig), history
    return w_out, None, history


def run_dataset(
    dataset_path: Path,
    output_dir: Path,
    max_iteration: int,
    plot_every: Optional[int],
    random_update: bool,
    use_pocket: bool,
    random_seed: int,
    save_animation: bool,
) -> tuple[np.ndarray, float, float]:
    data = np.load(dataset_path)
    variant = "pocket_mod" if use_pocket else "plain_pla"

    w, animation_bundle, history = pla(
        data=data,
        plot_every=plot_every,
        max_iteration=max_iteration,
        random_update=random_update,
        use_pocket=use_pocket,
        random_seed=random_seed,
    )

    train_acc = classification_accuracy(data["X"], data["Y"], w)
    test_acc = classification_accuracy(data["X_test"], data["Y_test"], w)

    print(
        f"[{dataset_path.stem}/{variant}] iterations={len(history)} "
        f"train_acc={train_acc:.3f} test_acc={test_acc:.3f}"
    )

    if save_animation and animation_bundle is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ani, fig = animation_bundle
        output_path = output_dir / f"{dataset_path.stem}_{variant}.gif"
        ani.save(output_path, writer=animation.PillowWriter(fps=2))
        plt.close(fig)
        print(f"Saved animation: {output_path}")

    return w, train_acc, test_acc


def main() -> None:
    # Task 1: Hoeffding bound.
    one_sided, two_sided = hoeffding_unhappy_probability_bound(
        n_samples=43,
        empirical_error=0.1,
        tolerated_error=0.2,
    )
    print("Hoeffding bound for P(E_out > 0.2 | E_in = 0.1, N = 43):")
    print(f"  one-sided <= {one_sided:.6f}")
    print(f"  two-sided <= {two_sided:.6f}")

    output_dir = Path("figures")

    # Task 2a+b: plain PLA on pla.npz.
    run_dataset(
        dataset_path=Path("data/pla.npz"),
        output_dir=output_dir,
        max_iteration=200,
        plot_every=1,
        random_update=False,
        use_pocket=False,
        random_seed=0,
        save_animation=True,
    )

    # Task 2c baseline: plain PLA on pocket.npz.
    run_dataset(
        dataset_path=Path("data/pocket.npz"),
        output_dir=output_dir,
        max_iteration=500,
        plot_every=1,
        random_update=False,
        use_pocket=False,
        random_seed=0,
        save_animation=True,
    )

    # Task 2c small modification:
    # random updates + pocket best weights.
    run_dataset(
        dataset_path=Path("data/pocket.npz"),
        output_dir=output_dir,
        max_iteration=500,
        plot_every=1,
        random_update=True,
        use_pocket=True,
        random_seed=63,
        save_animation=True,
    )


if __name__ == "__main__":
    main()
