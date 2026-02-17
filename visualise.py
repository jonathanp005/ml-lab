import matplotlib.pyplot as plt
import numpy as np

from typing import Optional

def visualise(w: np.ndarray,
              X: np.ndarray,
              Y: np.ndarray,
              ax=None,
              animated: bool = False,
              title: Optional[str] = None) -> tuple:
    """
    Visualises the dataset together with the separator

    Only works with 2D data and linear separator.

    Parameters
    ----------
    w : array-like, shape: (3,)
        the model weights, last element is the bias
    X : array-like, shape: (N, 2)
        the matrix of input elements
    Y : array-like an Nx1 matrix of labels
    ax : optional
        Matplotlib Axes to plot on. If omitted, a new figure will be created.
    animated : bool
        whether the plot will be used for animation
    title : str
        title of the plot

    Returns
    -------
    fig, ax :
        The figure and axes objects.
    """

    def separator_points(w):
        l = -w[2] / (w[0]**2 + w[1]**2)
        x0 = np.array([l * w[0], l * w[1]])
        v = np.array([-w[1], w[0]])
        v /= np.linalg.norm(v, ord=2)
        return np.stack([x0 + 5*v, x0 - 5*v])

    w = w.reshape((-1,))
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    artists = []
    positive_samples = X[Y == 1, :]
    negative_samples = X[Y == -1, :]
    artists += (
        ax.plot(
            positive_samples[:, 0],
            positive_samples[:, 1],
            'x',
            color='blue',
            label='$+1$',
            animated=animated))
    artists += (
        ax.plot(
            negative_samples[:, 0],
            negative_samples[:, 1],
            'o',
            color='orange',
            label='$-1$',
            animated=animated))
    if np.linalg.norm(w) < 1e-10 or np.linalg.norm(w[:2]) < 1e-10:
        print('Warning: the norm of the weight vector is to small to plot.')
    else:
        s = separator_points(w)
        artists += (
            ax.plot(
                s[:, 0],
                s[:, 1],
                '-',
                color='black',
                label='separator',
                animated=animated))
    artists.append(ax.legend(handles=artists))
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    if title:
        ax.set_title(title)
    ax.grid(True)
    return fig, ax, artists
