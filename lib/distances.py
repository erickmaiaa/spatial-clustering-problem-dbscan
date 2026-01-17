import numpy as np
import numpy.typing as npt


def euclidian_distance(p1: npt.NDArray[np.floating], p2: npt.NDArray[np.floating]) -> np.floating:
    return np.linalg.norm(p2 - p1)
