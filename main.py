""" 
TRABALHO DE INTELIGÊNCIA ARTIFICIAL

Equipe:
    - Agnaldo Erick Maia de Oliveira (539650) [ES]
    - Francisco Rodrigo de Santiago Pinheiro (554394) [ES]
    - Vitor Costa de Sousa (536678) [ES]
"""
import numpy as np
from lib.datasets import IRIS
from src.algorithms.dbscan import DBSCAN
from utils.plot import plot_clusters


def app() -> None:
    dbscan = DBSCAN(X=IRIS.data, eps=0.5, min_pts=5)
    classfication = dbscan.execute()
    plot_clusters(IRIS.data, np.array(classfication), dataset_ref="Iris")


if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        print('Errors:', e.args)
