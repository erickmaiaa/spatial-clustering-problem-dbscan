import numpy as np
import numpy.typing as npt
from collections import deque
from lib.distances import euclidian_distance


class DBSCAN:
    def __init__(self, X: npt.NDArray[np.floating], eps: float, min_pts: int) -> None:
        self.eps = np.float32(eps)
        self.min_pts = np.int32(min_pts)
        self.db = X

        points = np.int32(len(X))
        self.labels = np.full(points, -1)
        self.classfication = np.full(points, -1)
        self.visited = np.zeros(points, dtype=bool)

    def execute(self) -> deque[int]:
        cluster_id = 0

        for target_point in range(len(self.db)):
            if self.visited[target_point]:
                continue
            self.visited[target_point] = True
            neighbors = self.__range_query(np.int32(target_point))

            if len(neighbors) < self.min_pts:
                self.labels[target_point] = -1
                self.classfication[target_point] = -1
            else:
                self.__expand(
                    np.int32(target_point), neighbors, cluster_id,
                )
                cluster_id += 1
        return deque([int(n) for n in self.classfication])

    def __range_query(self, target_point: np.int32) -> deque[np.int32]:
        N: deque[np.int32] = deque()

        for i in range(len(self.db)):
            if euclidian_distance(self.db[target_point], self.db[i]) <= self.eps:
                N.append(np.int32(i))
        return N

    def __expand(self, target_point: np.int32, neighbors: deque[np.int32], cluster_id: int) -> None:
        self.labels[target_point] = cluster_id
        self.classfication[target_point] = 1
        i = 0

        while i < len(neighbors):
            target_neighbor = neighbors[i]

            if not self.visited[target_neighbor]:
                self.visited[target_neighbor] = True
                neighbor_neighbors = self.__range_query(target_neighbor)

                if len(neighbor_neighbors) >= self.min_pts:
                    self.classfication[target_neighbor] = 1

                    neighbors.extend(
                        n for n in neighbor_neighbors if n not in neighbors
                    )
                elif self.labels[target_neighbor] == -1:
                    self.classfication[target_neighbor] = 0

            if self.labels[target_neighbor] == -1:
                self.labels[target_neighbor] = cluster_id
            i += 1
