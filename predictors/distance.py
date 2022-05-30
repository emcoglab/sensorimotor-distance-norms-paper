from abc import ABC, abstractmethod

from numpy import array, corrcoef
from numpy.linalg import norm
from scipy.spatial.distance import minkowski, cosine as cosine_distance


class Distance(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def distance(self, u: array, v: array) -> float:
        raise NotImplementedError()


class Euclidean(Distance):
    @property
    def name(self): return "Euclidean"

    def distance(self, u: array, v: array) -> float:
        return norm(u - v)


class Cosine(Distance):
    @property
    def name(self): return "cosine"

    def distance(self, u: array, v: array) -> float:
        return cosine_distance(u, v)


class Correlation(Distance):
    @property
    def name(self): return "correlation"

    def distance(self, u: array, v: array) -> float:
        return 1 - corrcoef(u, v)[0, 1]


class Minkowski3(Distance):
    @property
    def name(self): return "Minkowski-3"

    def distance(self, u: array, v: array) -> float:
        return minkowski(u, v, p=3)
