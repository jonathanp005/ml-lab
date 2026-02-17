#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional
from visualise import visualise

# TODO

def plot_3d(X, Y):
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    ax1.scatter(X[:, 0], X[:, 1], Y, c='blue', label='train')
    ax1.scatter(X[:, 0], X[:, 1], Y, c='green', label='test')
    ax1.set_xlabel(f'$X_{0}$')
    ax1.set_ylabel(f'$X_{1}$')
    ax1.set_zlabel('$Y$')
    ax1.legend()
    ax2.scatter(X[:, 0], X[:, 2], Y, c='blue', label='train')
    ax2.scatter(X[:, 0], X[:, 2], Y, c='green', label='test')
    ax2.legend()
    ax2.set_xlabel(f'$X_{0}$')
    ax2.set_ylabel(f'$X_{2}$')
    ax2.set_zlabel('$Y$')
    return fig, (ax1, ax2)

# TODO

def pla(
    data,
    plot_every: Optional[int] = 1,
    max_iteration: int = 100
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, list]:
    # TODO
    if plot_every:
        fig, ax = plt.subplots()
        artists = []
    for i in range(max_iteration):
        # TODO
        acc = ...
        w, X, Y = ...
        if plot_every and i % plot_every == 0:
            _fig, _ax, artists_ = visualise(
                w,
                X,
                Y,
                ax=ax,
                animated=True,
                title=f'Training, accuracy: {acc:0.3}')
            artists.append(artists_)
    if plot_every:
        ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=1000)
        return w, (ani, fig)
    else:
        return w

def main():
    # TODO
    w, (ani, fig) = pla(...)
    plt.show()
    # TODO

if __name__ == "__main__":
    main()
