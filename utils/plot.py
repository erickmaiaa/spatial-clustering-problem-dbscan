import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from random import random


def plot_clusters(X: npt.NDArray[np.floating], classification: npt.NDArray[np.integer], dataset_ref: str) -> None:
    plt.figure(figsize=(8, 6))

    categories = {
        1: {"color": "red",
            "label": "Ponto Central (1)", "size": 30, "marker": "o"},
        0: {"color": "blue",
            "label": "Borda (0)", "size": 20, "marker": "o"},
        -1: {"color": "black",
             "label": "Ruído (-1)", "size": 10, "marker": "x"},
    }

    for point_type, props in categories.items():
        mask = classification == point_type

        if np.any(mask):
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                c=props["color"],
                s=props["size"],
                marker=props["marker"],
                label=props["label"],
                alpha=0.75
            )

    plt.title(f"Dataset: {dataset_ref}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis("equal")
    plt.legend()

    filename = f"./resources/output/fig-{int(random() * 10000)}.png"
    plt.savefig(filename, dpi=300)
