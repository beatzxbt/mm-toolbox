import numpy as np
from numba import njit

# This warning may be enabled in the future if it's function is still
# unclear. Currently, this is annoying for those who dont need linalg funcs.

# from warnings import warn as warning
# warning(
#     message="Numba optimized linalg functions will *only* work with float/complex type arrays.",
#     category=UserWarning
# )


@njit
def nbdot(A: np.ndarray, B: np.ndarray) -> float:
    return np.dot(A, B)


@njit
def nbnorm(A: np.ndarray, ord: int = None) -> float:
    return np.linalg.norm(A, ord)


@njit
def nbsolve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.linalg.solve(A, B)


@njit
def nbinv(A: np.ndarray) -> np.ndarray:
    return np.linalg.inv(A)


@njit
def nbdet(A: np.ndarray) -> float:
    return np.linalg.det(A)


@njit
def nbeig(A: np.ndarray):
    return np.linalg.eig(A)


@njit
def nbsvd(A: np.ndarray, full_matrices: bool):
    return np.linalg.svd(A, full_matrices)


@njit
def nbqr(A: np.ndarray):
    return np.linalg.qr(A)


@njit
def nbcholesky(A: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(A)
