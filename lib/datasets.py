from sklearn.datasets import make_circles, make_moons, load_iris

TWO_CIRCLES = make_circles(
    n_samples=300, factor=0.5, noise=0.05, random_state=42)

TWO_MOONS = make_moons(n_samples=300, noise=0.08, random_state=42)

IRIS = load_iris()
