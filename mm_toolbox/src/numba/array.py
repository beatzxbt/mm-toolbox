import numpy as np
from numba import njit
from numba.types import bool_
from typing import Tuple, Union

@njit
def nblinspace(start: float, stop: float, num: int = 50, endpoint: bool = True) -> np.ndarray:
    return np.linspace(start, stop, num, endpoint)

@njit
def nbgeomspace(start: float, stop: float, num: int = 50, endpoint: bool = True) -> np.ndarray:
    return np.geomspace(start, stop, num, endpoint)

@njit
def nbarange(start: float, stop: float = None, step: float = 1) -> np.ndarray:
    return np.arange(start, stop, step)

@njit
def nblogspace(start: float, stop: float, num: int = 50, endpoint: bool = True, base: float = 10.0) -> np.ndarray:
    return np.logspace(start, stop, num, endpoint, base)

@njit
def nbzeros(shape: Union[int, Tuple[int, ...]], dtype: np.dtype = np.float64) -> np.ndarray:
    return np.zeros(shape, dtype)

@njit
def nbones(shape: Union[int, Tuple[int, ...]], dtype: np.dtype = np.float64) -> np.ndarray:
    return np.ones(shape, dtype)

@njit
def nbfull(shape: Union[int, Tuple[int, ...]], fill_value: float, dtype: np.dtype = np.float64) -> np.ndarray:
    return np.full(shape, fill_value, dtype)

@njit
def nbeye(N: int, M: int = None, k: int = 0, dtype: np.dtype = np.float64) -> np.ndarray:
    return np.eye(N, M, k, dtype)

@njit
def nbdiag(v: np.ndarray, k: int = 0) -> np.ndarray:
    return np.diag(v, k)

@njit(inline="always")
def nbisin(a: np.ndarray, b: np.ndarray) -> np.ndarray[bool]:
    out_len = a.size
    out = np.empty(out_len, dtype=bool_)
    b_set = set(b)
    
    for i in range(out_len):
        out[i] = a[i] in b_set

    return out

@njit
def nbwhere(condition, x=None, y=None) -> np.ndarray:
    return np.where(condition, x, y)

@njit
def nbdiff(a: np.ndarray, n: int = 1) -> np.ndarray:
    return np.diff(a, n)

@njit
def nbflip(a: np.ndarray) -> np.ndarray:
    return np.flip(a)

@njit
def nbsort(a: np.ndarray) -> np.ndarray:
    return np.sort(a)

@njit
def nbargsort(a: np.ndarray) -> np.ndarray:
    return np.argsort(a)

@njit
def nbconcatenate(arrays: Tuple[np.ndarray, ...], axis: int = 0) -> np.ndarray:
    return np.concatenate(arrays, axis)

@njit
def nbravel(a: np.ndarray) -> np.ndarray:
    return np.ravel(a)

@njit
def nbreshape(a: np.ndarray, newshape: Tuple[int, ...]) -> np.ndarray:
    return np.reshape(a, newshape)

@njit
def nbtranspose(a: np.ndarray) -> np.ndarray:
    return np.transpose(a)

@njit
def nbhstack(tup: Tuple[np.ndarray, ...]) -> np.ndarray:
    return np.hstack(tup)

@njit
def nbvstack(tup: Tuple[np.ndarray, ...]) -> np.ndarray:
    return np.vstack(tup)

@njit
def nbclip(a: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    return np.clip(a, a_min, a_max)

@njit
def nbunique(a: np.ndarray) -> np.ndarray:
    return np.unique(a)

@njit
def nbtile(A: np.ndarray, reps: Union[int, Tuple[int, ...]]) -> np.ndarray:
    return np.tile(A, reps)

@njit
def nbrepeat(a: np.ndarray, repeats: Union[int, np.ndarray], axis: int = None) -> np.ndarray:
    return np.repeat(a, repeats, axis)

@njit
def nbstack(arrays: Tuple[np.ndarray, ...], axis: int = 0) -> np.ndarray:
    return np.stack(arrays, axis)

@njit
def nbroll(a: np.ndarray, shift: int, axis: int = None) -> np.ndarray:
    return np.roll(a, shift, axis)
