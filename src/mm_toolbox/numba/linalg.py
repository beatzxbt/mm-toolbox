from importlib.util import find_spec

if find_spec("scipy") is None:
    raise ImportError(
        "SciPy is required for all linalg functions, do 'pip install scipy'."
    )

###

import numpy as np
from numba import njit
from typing import Tuple, Union


@njit(inline="always")
def nbcholesky(a: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(a)


@njit(inline="always")
def nbcond(a: np.ndarray, p: Union[None, int, float, str] = None) -> float:
    return np.linalg.cond(a, p=p)


@njit(inline="always")
def nbcov(
    m: np.ndarray,
    y: np.ndarray = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Union[int, None] = None,
) -> np.ndarray:
    """
    Constraints:
    * 'm', 'y': 2D arrays
    """
    if m.ndim != 2 or (y is not None and y.ndim != 2):
        raise ValueError("Covariance matrices must be 2D.")
    return np.cov(m, y, rowvar=rowvar, bias=bias, ddof=ddof)


@njit(inline="always")
def nbdet(a: np.ndarray) -> float:
    return np.linalg.det(a)


@njit(inline="always")
def nbdot(A: np.ndarray, B: np.ndarray) -> float:
    return np.dot(A, B)


@njit(inline="always")
def nbeig(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.linalg.eig(a)


@njit(inline="always")
def nbeigh(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.linalg.eigh(a)


@njit(inline="always")
def nbeigvals(a: np.ndarray) -> np.ndarray:
    return np.linalg.eigvals(a)


@njit(inline="always")
def nbeigvalsh(a: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(a)


@njit(inline="always")
def nbinv(a: np.ndarray) -> np.ndarray:
    return np.linalg.inv(a)


@njit(inline="always")
def nbkron(a: np.ndarray, b: np.ndarray, order: str = "C") -> np.ndarray:
    return np.kron(a, b)


@njit(inline="always")
def nblstsq(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return np.linalg.lstsq(a, b)


@njit(inline="always")
def nbmatrix_power(a: np.ndarray, n: int) -> np.ndarray:
    return np.linalg.matrix_power(a, n)


@njit(inline="always")
def nbmatrix_rank(a: np.ndarray) -> int:
    return np.linalg.matrix_rank(a)


@njit(inline="always")
def nbnorm(a: np.ndarray, ord: Union[int, None] = None) -> float:
    return np.linalg.norm(a, ord=ord)


@njit(inline="always")
def nbouter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.outer(a, b)


@njit(inline="always")
def nbpinv(a: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(a)


@njit(inline="always")
def nbqr(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.linalg.qr(a)


@njit(inline="always")
def nbslodet(a: np.ndarray) -> Tuple[float, float]:
    return np.linalg.slogdet(a)


@njit(inline="always")
def nbsolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(a, b)


@njit(inline="always")
def nbsvd(
    a: np.ndarray, full_matrices: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.linalg.svd(a, full_matrices=full_matrices)


@njit(inline="always")
def nbtrace(a: np.ndarray) -> np.ndarray:
    return np.trace(a)


@njit(inline="always")
def nbvdot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.vdot(a, b)
