from pathlib import Path

from numpy import load, save, cov, array

_cov_matrix_name = "covariance_matrix.npy"


def load_covariance_matrix(from_dir: Path) -> array:
    path = Path(from_dir, _cov_matrix_name)
    return load(path.as_posix())


def save_covariance_matrix(input_matrix, to_dir: Path, overwrite: bool = False) -> array:
    """input_matrix.shape = (dims, observations)"""
    path = Path(to_dir, _cov_matrix_name)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    covariance_matrix = _compute_covariance_matrix(input_matrix)
    with path.open("wb") as f:
        save(f, covariance_matrix)
    return covariance_matrix


def _compute_covariance_matrix(input_matrix) -> array:
    return cov(input_matrix)
