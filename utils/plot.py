import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from collections import deque
from random import random


def plot_clusters(X: npt.NDArray[np.floating], labels: deque[int]) -> None:
    unique_labels = np.unique(labels)

    plt.figure(figsize=(6, 6))

    for label in unique_labels:
        mask = labels == label

        if label == -1:
            # Ruído
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                c="black",
                s=10,
                label="Noise"
            )
        else:
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                s=10,
                label=f"Cluster {label}"
            )

    plt.title("DBSCAN implementado do zero")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.savefig(f"./resources/output/fig-{int(random() * 10000)}.png", dpi=300)
