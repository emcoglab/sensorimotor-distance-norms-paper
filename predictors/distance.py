from abc import ABC, abstractmethod

from numpy import array, corrcoef
from numpy.linalg import norm, inv
from scipy.spatial.distance import minkowski, cosine as cosine_distance, mahalanobis


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


class Mahalanobis(Distance):
    @property
    def name(self): return "Mahalanobis"

    def __init__(self, with_covariance_matrix: array):
        super().__init__()
        self.covariance_matrix: array = with_covariance_matrix
        self._cov_inv: array = inv(with_covariance_matrix)
        assert self._cov_inv.shape[1] == self.dimensionality

    @property
    def dimensionality(self) -> int:
        return self.covariance_matrix.shape[0]

    def distance(self, u: array, v: array) -> float:
        return mahalanobis(u, v, self._cov_inv)
