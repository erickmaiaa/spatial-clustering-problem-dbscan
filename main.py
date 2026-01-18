""" 
TRABALHO DE INTELIGÊNCIA ARTIFICIAL

Equipe:
    - Agnaldo Erick Maia de Oliveira (539650) [ES]
    - Francisco Rodrigo de Santiago Pinheiro (554394) [ES]
    - Vitor Costa de Sousa (536678) [ES]
"""
from lib.datasets import TWO_CIRCLES
from src.algorithms.dbscan import DBSCAN
from utils.plot import plot_clusters


def app() -> None:
    instance1 = DBSCAN(X=TWO_CIRCLES, eps=0.2, min_pts=5)
    labels = instance1.execute()
    plot_clusters(TWO_CIRCLES, labels)


if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        print('Errors:', e.args)
