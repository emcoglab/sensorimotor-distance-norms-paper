from abc import ABC, abstractmethod

from numpy import array, corrcoef
from numpy.linalg import norm
from scipy.spatial.distance import minkowski, cosine as cosine_distance


class Distance(ABC):
    @classmethod
    @property
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError()

    @abstractmethod
    def distance(self, u: array, v: array) -> float:
        raise NotImplementedError()


class Euclidean(Distance):
    @classmethod
    @property
    def name(cls): return "Euclidean"

    def distance(self, u: array, v: array) -> float:
        return norm(u - v)


class Cosine(Distance):
    @classmethod
    @property
    def name(cls): return "cosine"

    def distance(self, u: array, v: array) -> float:
        return cosine_distance(u, v)


class Correlation(Distance):
    @classmethod
    @property
    def name(cls): return "correlation"

    def distance(self, u: array, v: array) -> float:
        return 1 - corrcoef(u, v)[0, 1]


class Minkowski3(Distance):
    @classmethod
    @property
    def name(cls): return "Minkowski-3"

    def distance(self, u: array, v: array) -> float:
        return minkowski(u, v, p=3)
