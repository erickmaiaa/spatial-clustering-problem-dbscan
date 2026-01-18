import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from collections import deque
from random import random


def plot_clusters(X: npt.NDArray[np.floating], point_types: npt.NDArray[np.integer]) -> None:
    plt.figure(figsize=(8, 6))

    categories = [
        {"value": 1, "color": "red", "label": "Ponto Central (1)", "size": 30, "marker": "o"},
        {"value": 0, "color": "blue", "label": "Borda (0)", "size": 20, "marker": "o"},
        {"value": -1, "color": "gray", "label": "Ruído (-1)", "size": 10, "marker": "x"},
    ]

    for cat in categories:
        mask = (point_types == cat["value"])
        
        if np.any(mask):
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                c=cat["color"],
                s=cat["size"],
                marker=cat["marker"],
                label=cat["label"],
                alpha=0.7
            )

    plt.title("DBSCAN: Classificação por Tipos de Pontos")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis("equal")
    
    filename = f"./resources/output/plot-types-{int(random() * 10000)}.png"
    plt.savefig(filename, dpi=300)