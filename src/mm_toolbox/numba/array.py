import numpy as np
from numba import njit
from numba.types import bool_
from typing import Tuple, Union, List


@njit(inline="always")
def nballclose(
    a: np.ndarray, b: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    return np.allclose(a, b, rtol=rtol, atol=atol)


@njit(inline="always")
def nbappend(arr: np.ndarray, values: np.ndarray, axis: int = None) -> np.ndarray:
    return np.append(arr, values, axis=axis)


@njit(inline="always")
def nbarange(start: float, stop: float = None, step: float = 1) -> np.ndarray:
    """
    Constraints:
    * 'start', 'stop', 'step': float or int
    """
    return np.arange(start, stop, step)


@njit(inline="always")
def nbargsort(a: np.ndarray):
    return np.argsort(a)


@njit(inline="always")
def nbaround(a: np.ndarray, decimals: int = 0) -> np.ndarray:
    return np.around(a, decimals=decimals)


@njit(inline="always")
def nbarray_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(a, b)


@njit(inline="always")
def nbarray_split(
    ary: np.ndarray, indices_or_sections: Union[int, List[int]], axis: int = 0
) -> List[np.ndarray]:
    """
    Constraints:
    * 'indices_or_sections': int or list of ints
    """
    return np.array_split(ary, indices_or_sections, axis=axis)


@njit(inline="always")
def nbasarray(a: Union[List, Tuple, np.ndarray], dtype: np.dtype = None) -> np.ndarray:
    return np.asarray(a, dtype=dtype)


@njit(inline="always")
def nbbroadcast_arrays(*args: np.ndarray) -> List[np.ndarray]:
    return np.broadcast_arrays(*args)


@njit(inline="always")
def nbclip(a: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    return np.clip(a, a_min, a_max)


@njit(inline="always")
def nbcolumn_stack(tup: Tuple[np.ndarray, ...]) -> np.ndarray:
    return np.column_stack(tup)


@njit(inline="always")
def nbconcatenate(arrays: Tuple[np.ndarray, ...]) -> np.ndarray:
    return np.concatenate(arrays)


@njit(inline="always")
def nbconvolve(a: np.ndarray, v: np.ndarray, mode: str = "full") -> np.ndarray:
    """
    Constraints:
    * 'a', 'v': 1D arrays
    * 'mode': ['full', 'valid', 'same']
    """
    if a.ndim != 1 or v.ndim != 1:
        raise ValueError("Convolution arrays must be 1D.")
    return np.convolve(a, v, mode=mode)


@njit(inline="always")
def nbcopy(a: np.ndarray) -> np.ndarray:
    return np.copy(a)


@njit(inline="always")
def nbcorrelate(a: np.ndarray, v: np.ndarray, mode: str = "valid") -> np.ndarray:
    """
    Constraints:
    * 'a', 'v': 1D arrays
    * 'mode': ['full', 'valid', 'same']
    """
    if a.ndim != 1 or v.ndim != 1:
        raise ValueError("Correlation arrays must be 1D.")
    return np.correlate(a, v, mode=mode)


@njit(inline="always")
def nbcount_nonzero(a: np.ndarray) -> int:
    return np.count_nonzero(a)


@njit(inline="always")
def nbcross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Constraints:
    * 'a', 'b': 1D or 2D arrays
    * One of the dimensions must be 3
    """
    if a.shape[-1] not in (2, 3) and b.shape[-1] not in (2, 3):
        raise ValueError("Cross product requires at least one array with dimension 3.")
    return np.cross(a, b)


@njit(inline="always")
def nbdiag(v: np.ndarray, k: int = 0) -> np.ndarray:
    return np.diag(v, k)


@njit(inline="always")
def nbdiagflat(v: np.ndarray, k: int = 0) -> np.ndarray:
    return np.diagflat(v, k)


@njit(inline="always")
def nbdiff(a: np.ndarray, n: int = 1) -> np.ndarray:
    """
    Constraints:
    * 'a': 1D array
    * 'n': int >= 0
    """
    assert n >= 0 and a.ndim == 1

    if n == 0:
        return a.copy()

    a_size = a.size
    out_size = max(a_size - n, 0)
    out = np.empty(out_size, dtype=a.dtype)

    if out_size == 0:
        return out

    work = np.empty_like(a)

    # First iteration: diff a into work
    for i in range(a_size - 1):
        work[i] = a[i + 1] - a[i]

    # Other iterations: diff work into itself
    for niter in range(1, n):
        for i in range(a_size - niter - 1):
            work[i] = work[i + 1] - work[i]

    # Copy final diff into out
    out[:] = work[:out_size]

    return out


@njit(inline="always")
def nbdigitize(x: np.ndarray, bins: np.ndarray, right: bool = False) -> np.ndarray:
    return np.digitize(x, bins, right=right)


@njit(inline="always")
def nbdstack(tup: Tuple[np.ndarray, ...]) -> np.ndarray:
    return np.dstack(tup)


@njit(inline="always")
def nbediff1d(
    ary: np.ndarray,
    to_end: Union[np.ndarray, None] = None,
    to_begin: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """
    Constraints:
    * 'ary': 1D array
    """
    if ary.ndim != 1:
        raise ValueError("ediff1d requires 1D array.")
    return np.ediff1d(ary, to_end=to_end, to_begin=to_begin)


@njit(inline="always")
def nbexpand_dims(a: np.ndarray, axis: int) -> np.ndarray:
    return np.expand_dims(a, axis)


@njit(inline="always")
def nbextract(condition: np.ndarray, arr: np.ndarray) -> np.ndarray:
    return np.extract(condition, arr)


@njit(inline="always")
def nbeye(
    N: int, M: int = None, k: int = 0, dtype: np.dtype = np.float64
) -> np.ndarray:
    return np.eye(N, M, k, dtype)


@njit(inline="always")
def nbfill_diagonal(a: np.ndarray, val: float, wrap: bool = False) -> None:
    return np.fill_diagonal(a, val, wrap=wrap)


@njit(inline="always")
def nbflatten(a: np.ndarray) -> np.ndarray:
    return a.flatten()


@njit(inline="always")
def nbflatnonzero(a: np.ndarray) -> np.ndarray:
    return np.flatnonzero(a)


@njit(inline="always")
def nbflip(a: np.ndarray) -> np.ndarray:
    return np.flip(a)


@njit(inline="always")
def nbflipud(a: np.ndarray) -> np.ndarray:
    return np.flipud(a)


@njit(inline="always")
def nbfliplr(a: np.ndarray) -> np.ndarray:
    return np.fliplr(a)


@njit(inline="always")
def nbfull(
    shape: Union[int, Tuple[int, ...]], fill_value: float, dtype: np.dtype = np.float64
) -> np.ndarray:
    return np.full(shape, fill_value, dtype)


@njit(inline="always")
def nbfull_like(a: np.ndarray, fill_value: float, dtype: np.dtype = None) -> np.ndarray:
    return np.full_like(a, fill_value, dtype=dtype)


@njit(inline="always")
def nbgeomspace(start: float, stop: float, num: int = 50) -> np.ndarray:
    return np.geomspace(start, stop, num)


@njit(inline="always")
def nbhistogram(
    a: np.ndarray, bins: Union[int, np.ndarray] = 10, range: Tuple[float, float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    return np.histogram(a, bins=bins, range=range)


@njit(inline="always")
def nbhsplit(
    ary: np.ndarray, indices_or_sections: Union[int, Tuple[int, ...]]
) -> Tuple[np.ndarray, ...]:
    return np.hsplit(ary, indices_or_sections)


@njit(inline="always")
def nbhstack(tup: Tuple[np.ndarray, ...]) -> np.ndarray:
    return np.hstack(tup)


@njit(inline="always")
def nbidentity(n: int, dtype: np.dtype = float) -> np.ndarray:
    return np.identity(n, dtype=dtype)


@njit(inline="always")
def nbindices(dimensions: Tuple[int, ...]) -> np.ndarray:
    return np.indices(dimensions)


@njit(inline="always")
def nbinterp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.interp(x, xp, fp)


@njit(inline="always")
def nbintersect1d(ar1: np.ndarray, ar2: np.ndarray) -> np.ndarray:
    return np.intersect1d(ar1, ar2)


@njit(inline="always")
def nbisclose(
    a: np.ndarray,
    b: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> np.ndarray:
    return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@njit(inline="always")
def nbiscomplex(x: np.ndarray) -> np.ndarray:
    return np.iscomplex(x)


@njit(inline="always")
def nbiscomplexobj(x: np.ndarray) -> bool:
    return np.iscomplexobj(x)


@njit(inline="always")
def nbisin(a: np.ndarray, b: np.ndarray) -> np.ndarray[bool]:
    """
    Constaints:
    * 'a': dim=1, dtype='same as other'
    * 'b': dim=1, dtype='same as other'
    """
    assert a.ndim == 1 and b.ndim == 1, "2D arrays not supported."

    b_set = set(b)
    out_len = a.size
    out = np.empty(out_len, dtype=bool_)
    for i in range(out_len):
        out[i] = a[i] in b_set

    return out


@njit(inline="always")
def nbisneginf(x: np.ndarray) -> np.ndarray:
    return np.isneginf(x)


@njit(inline="always")
def nbisposinf(x: np.ndarray) -> np.ndarray:
    return np.isposinf(x)


@njit(inline="always")
def nbisreal(x: np.ndarray) -> np.ndarray:
    return np.isreal(x)


@njit(inline="always")
def nbisrealobj(x: np.ndarray) -> bool:
    return np.isrealobj(x)


@njit(inline="always")
def nbisscalar(num: Union[int, float, complex, bool, np.generic]) -> bool:
    return np.isscalar(num)


@njit(inline="always")
def nbkaiser(M: int, beta: float) -> np.ndarray:
    return np.kaiser(M, beta)


@njit(inline="always")
def nbnan_to_num(x: np.ndarray, copy: bool = True, nan: float = 0.0) -> np.ndarray:
    return np.nan_to_num(x, copy=copy, nan=nan)


@njit(inline="always")
def nbones(
    shape: Union[int, Tuple[int, ...]], dtype: np.dtype = np.float64
) -> np.ndarray:
    return np.ones(shape, dtype)


@njit(inline="always")
def nbpartition(a: np.ndarray, kth: Union[int, Tuple[int, ...]]) -> np.ndarray:
    return np.partition(a, kth)


@njit(inline="always")
def nbptp(a: np.ndarray) -> np.ndarray:
    return np.ptp(a)


@njit(inline="always")
def nbrepeat(a: np.ndarray, repeats: Union[int, np.ndarray]) -> np.ndarray:
    return np.repeat(a, repeats)


@njit(inline="always")
def nbreshape(a: np.ndarray, newshape: Tuple[int, ...]) -> np.ndarray:
    return np.reshape(a, newshape)


@njit(inline="always")
def nbroll(a: np.ndarray, shift: int, axis: int) -> np.ndarray:
    """
    Constraints:
    * 'axis': int >= 0
    """
    assert axis >= 0, "Axis must be positive."

    if shift == 0:
        return a

    if a.ndim == 1:
        if shift > 0:
            return np.concat((a[-shift:], a[:-shift]))
        else:
            return np.concat((a[shift:], a[:shift]))

    # Numba throws index error without this. Seems that it cant
    # infer the early return from ndim==1 branch and fails to
    # generate the memory map correctly for 'a'.
    assert a.ndim > 1

    shift = shift % a.shape[axis]

    out = np.empty_like(a)
    axis_len = a.shape[axis]

    # Roll the array
    for i in range(axis_len):
        new_index = (i + shift) % axis_len
        if axis == 0:
            out[new_index, :] = a[i, :]
        else:
            out[:, new_index] = a[:, i]

    return out


@njit(inline="always")
def nbrot90(m: np.ndarray, k: int = 1) -> np.ndarray:
    return np.rot90(m, k=k)


@njit(inline="always")
def nbravel(a: np.ndarray) -> np.ndarray:
    return np.ravel(a)


@njit(inline="always")
def nbrow_stack(tup: Tuple[np.ndarray, ...]) -> np.ndarray:
    return np.row_stack(tup)


@njit(inline="always")
def nbround(a: np.ndarray, decimals: int = 0, out: np.ndarray = None) -> np.ndarray:
    return np.round(a, decimals=decimals, out=out)


@njit(inline="always")
def nbsearchsorted(
    a: np.ndarray, v: Union[int, float, np.ndarray], side: str = "left"
) -> np.ndarray:
    return np.searchsorted(a, v, side=side)


@njit(inline="always")
def nbselect(
    condlist: List[np.ndarray],
    choicelist: List[np.ndarray],
    default: Union[int, float] = 0,
) -> np.ndarray:
    return np.select(condlist, choicelist, default=default)


@njit(inline="always")
def nbshape(a: np.ndarray) -> Tuple[int, ...]:
    return a.shape


@njit(inline="always")
def nbsort(a: np.ndarray) -> np.ndarray:
    return np.sort(a)


@njit(inline="always")
def nbsplit(
    ary: np.ndarray, indices_or_sections: Union[int, np.ndarray], axis: int = 0
) -> List[np.ndarray]:
    return np.split(ary, indices_or_sections, axis=axis)


@njit(inline="always")
def nbstack(arrays: Tuple[np.ndarray, ...], axis: int = 0) -> np.ndarray:
    return np.stack(arrays, axis=axis)


@njit(inline="always")
def nbswapaxes(a: np.ndarray, axis1: int, axis2: int) -> np.ndarray:
    return np.swapaxes(a, axis1, axis2)


@njit(inline="always")
def nbtake(a: np.ndarray, indices: Union[int, np.ndarray]) -> np.ndarray:
    return np.take(a, indices)


@njit(inline="always")
def nbtranspose(a: np.ndarray, axes: Tuple[int, ...] = None) -> np.ndarray:
    return np.transpose(a, axes=axes)


@njit(inline="always")
def nbtri(N: int, M: int = None, k: int = 0) -> np.ndarray:
    if not isinstance(k, int):
        raise TypeError("'k' must be an Integer.")
    return np.tri(N, M, k=k)


@njit(inline="always")
def nbtril(m: np.ndarray, k: int = 0) -> np.ndarray:
    return np.tril(m, k=k)


@njit(inline="always")
def nbtril_indices(n: int, k: int = 0, m: int = None) -> Tuple[np.ndarray, np.ndarray]:
    return np.tril_indices(n, k=k, m=m)


@njit(inline="always")
def nbtril_indices_from(arr: np.ndarray, k: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    return np.tril_indices_from(arr, k=k)


@njit(inline="always")
def nbtriu(m: np.ndarray, k: int = 0) -> np.ndarray:
    return np.triu(m, k=k)


@njit(inline="always")
def nbtriu_indices(n: int, k: int = 0, m: int = None) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=k, m=m)


@njit(inline="always")
def nbtriu_indices_from(arr: np.ndarray, k: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices_from(arr, k=k)


@njit(inline="always")
def nbtrim_zeros(filt: np.ndarray, trim: str = "fb") -> np.ndarray:
    return np.trim_zeros(filt, trim=trim)


@njit(inline="always")
def nbunion1d(ar1: np.ndarray, ar2: np.ndarray) -> np.ndarray:
    return np.union1d(ar1, ar2)


@njit(inline="always")
def nbunique(a: np.ndarray) -> np.ndarray:
    return np.unique(a)


@njit(inline="always")
def nbunwrap(p: np.ndarray, discont: float = np.pi, axis: int = -1) -> np.ndarray:
    return np.unwrap(p, discont=discont, axis=axis)


@njit(inline="always")
def nbvander(x: np.ndarray, N: int = None, increasing: bool = False) -> np.ndarray:
    return np.vander(x, N=N, increasing=increasing)


@njit(inline="always")
def nbvsplit(
    ary: np.ndarray, indices_or_sections: Union[int, np.ndarray]
) -> List[np.ndarray]:
    return np.vsplit(ary, indices_or_sections)


@njit(inline="always")
def nbvstack(tup: Tuple[np.ndarray, ...]) -> np.ndarray:
    return np.vstack(tup)


@njit(inline="always")
def nbwhere(
    condition: np.ndarray, x: np.ndarray = None, y: np.ndarray = None
) -> np.ndarray:
    return np.where(condition, x, y)


@njit(inline="always")
def nbzeros(shape: Union[int, Tuple[int, ...]], dtype: np.dtype = float) -> np.ndarray:
    return np.zeros(shape, dtype)


@njit(inline="always")
def nbzeros_like(a: np.ndarray, dtype: np.dtype = None) -> np.ndarray:
    return np.zeros_like(a, dtype=dtype)


@njit(inline="always")
def nblinspace(start: float, stop: float, num: int = 50) -> np.ndarray:
    return np.linspace(start, stop, num)


@njit(inline="always")
def nblogspace(start: float, stop: float, num: int = 50) -> np.ndarray:
    return np.logspace(start, stop, num)
