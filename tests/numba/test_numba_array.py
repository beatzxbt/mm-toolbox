import unittest
import numpy as np
from mm_toolbox.numba.array import (
    nballclose as nballclose,
    nbappend as nbappend,
    nbargsort as nbargsort,
    nbarange as nbarange,
    nbaround as nbaround,
    nbarray_equal as nbarray_equal,
    nbarray_split as nbarray_split,
    nbasarray as nbasarray,
    nbbroadcast_arrays as nbbroadcast_arrays,
    nbclip as nbclip,
    nbcolumn_stack as nbcolumn_stack,
    nbconcatenate as nbconcatenate,
    nbconvolve as nbconvolve,
    nbcopy as nbcopy,
    nbcorrelate as nbcorrelate,
    nbcount_nonzero as nbcount_nonzero,
    nbcross as nbcross,
    nbdiff as nbdiff,
    nbdigitize as nbdigitize,
    nbdiag as nbdiag,
    nbdiagflat as nbdiagflat,
    nbdstack as nbdstack,
    nbediff1d as nbediff1d,
    nbexpand_dims as nbexpand_dims,
    nbextract as nbextract,
    nbeye as nbeye,
    nbfill_diagonal as nbfill_diagonal,
    nbflatten as nbflatten,
    nbflatnonzero as nbflatnonzero,
    nbflip as nbflip,
    nbfliplr as nbfliplr,
    nbflipud as nbflipud,
    nbfull as nbfull,
    nbfull_like as nbfull_like,
    nbgeomspace as nbgeomspace,
    nbhistogram as nbhistogram,
    nbhsplit as nbhsplit,
    nbhstack as nbhstack,
    nbidentity as nbidentity,
    nbindices as nbindices,
    nbinterp as nbinterp,
    nbintersect1d as nbintersect1d,
    nbisclose as nbisclose,
    nbiscomplex as nbiscomplex,
    nbiscomplexobj as nbiscomplexobj,
    nbisin as nbisin,
    nbisneginf as nbisneginf,
    nbisposinf as nbisposinf,
    nbisreal as nbisreal,
    nbisrealobj as nbisrealobj,
    nbisscalar as nbisscalar,
    nbkaiser as nbkaiser,
    nblinspace as nblinspace,
    nblogspace as nblogspace,
    nbnan_to_num as nbnan_to_num,
    nbones as nbones,
    nbpartition as nbpartition,
    nbptp as nbptp,
    nbrepeat as nbrepeat,
    nbreshape as nbreshape,
    nbroll as nbroll,
    nbrot90 as nbrot90,
    nbravel as nbravel,
    nbrow_stack as nbrow_stack,
    nbround as nbround,
    nbsearchsorted as nbsearchsorted,
    nbselect as nbselect,
    nbshape as nbshape,
    nbsort as nbsort,
    nbsplit as nbsplit,
    nbstack as nbstack,
    nbswapaxes as nbswapaxes,
    nbtake as nbtake,
    nbtranspose as nbtranspose,
    nbtri as nbtri,
    nbtril as nbtril,
    nbtril_indices as nbtril_indices,
    nbtril_indices_from as nbtril_indices_from,
    nbtriu as nbtriu,
    nbtriu_indices as nbtriu_indices,
    nbtriu_indices_from as nbtriu_indices_from,
    nbtrim_zeros as nbtrim_zeros,
    nbunion1d as nbunion1d,
    nbunique as nbunique,
    nbunwrap as nbunwrap,
    nbvander as nbvander,
    nbvsplit as nbvsplit,
    nbvstack as nbvstack,
    nbwhere as nbwhere,
    nbzeros as nbzeros,
    nbzeros_like as nbzeros_like,
)


class TestNumbaFuncs(unittest.TestCase):
    def test_nballclose(self):
        np_result = np.allclose(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        nb_result = nballclose(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        self.assertEqual(nb_result, np_result)

        np_result = np.allclose(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.1, 3.0]), rtol=1e-3
        )
        nb_result = nballclose(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.1, 3.0]), rtol=1e-3
        )
        self.assertEqual(nb_result, np_result)

    def test_nbappend(self):
        np_result = np.append(np.array([1, 2, 3]), np.array([4, 5, 6]))
        nb_result = nbappend(np.array([1, 2, 3]), np.array([4, 5, 6]))
        np.testing.assert_array_equal(nb_result, np_result)

        np_result = np.append(np.array([[1, 2], [3, 4]]), np.array([[5, 6]]), axis=0)
        nb_result = nbappend(np.array([[1, 2], [3, 4]]), np.array([[5, 6]]), axis=0)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbarange(self):
        np_result = np.arange(0.0, 10.0, 2.0)
        nb_result = nbarange(0.0, 10.0, 2.0)
        np.testing.assert_array_equal(nb_result, np_result)

        np_result = np.arange(5.0)
        nb_result = nbarange(5.0)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbargsort(self):
        np_result = np.argsort(np.array([3, 1, 2]))
        nb_result = nbargsort(np.array([3, 1, 2]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbaround(self):
        np_result = np.around(np.array([1.1234, 2.5678]), decimals=2)
        nb_result = nbaround(np.array([1.1234, 2.5678]), decimals=2)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbarray_equal(self):
        np_result = np.array_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
        nb_result = nbarray_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
        self.assertEqual(nb_result, np_result)

    def test_nbarray_split(self):
        np_result = np.array_split(np.array([1, 2, 3, 4, 5]), 2)
        nb_result = nbarray_split(np.array([1, 2, 3, 4, 5]), 2)
        for nb_array, np_array in zip(nb_result, np_result):
            np.testing.assert_array_equal(nb_array, np_array)

    def test_nbasarray(self):
        np_result = np.asarray([1, 2, 3])
        nb_result = nbasarray([1, 2, 3])
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbbroadcast_arrays(self):
        np_result = np.broadcast_arrays(np.array([1, 2, 3]), np.array([[1], [2], [3]]))
        nb_result = nbbroadcast_arrays(np.array([1, 2, 3]), np.array([[1], [2], [3]]))
        for nb_array, np_array in zip(nb_result, np_result):
            np.testing.assert_array_equal(nb_array, np_array)

    def test_nbclip(self):
        np_result = np.clip(np.array([1, 2, 3, 4, 5]), 2, 4)
        nb_result = nbclip(np.array([1, 2, 3, 4, 5]), 2, 4)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbcolumn_stack(self):
        np_result = np.column_stack((np.array([1, 2]), np.array([3, 4])))
        nb_result = nbcolumn_stack((np.array([1, 2]), np.array([3, 4])))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbconcatenate(self):
        np_result = np.concatenate((np.array([1, 2]), np.array([3, 4])))
        nb_result = nbconcatenate((np.array([1, 2]), np.array([3, 4])))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbconvolve(self):
        np_result = np.convolve(np.array([1, 2, 3]), np.array([0, 1, 0.5]), "same")
        nb_result = nbconvolve(np.array([1, 2, 3]), np.array([0, 1, 0.5]), "same")
        np.testing.assert_array_equal(nb_result, np_result)

        with self.assertRaises(ValueError):
            nb_result = nbconvolve(np.array([[1, 2], [3, 4]]), np.array([1, 2]), "same")

    def test_nbcopy(self):
        np_result = np.copy(np.array([1, 2, 3]))
        nb_result = nbcopy(np.array([1, 2, 3]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbcorrelate(self):
        np_result = np.correlate(np.array([1, 2, 3]), np.array([0, 1, 0.5]), "same")
        nb_result = nbcorrelate(np.array([1, 2, 3]), np.array([0, 1, 0.5]), "same")
        np.testing.assert_array_equal(nb_result, np_result)

        with self.assertRaises(ValueError):
            nb_result = nbcorrelate(
                np.array([[1, 2], [3, 4]]), np.array([1, 2]), "same"
            )

    def test_nbcount_nonzero(self):
        np_result = np.count_nonzero(np.array([1, 0, 2, 0, 3, 0]))
        nb_result = nbcount_nonzero(np.array([1, 0, 2, 0, 3, 0]))
        self.assertEqual(nb_result, np_result)

    def test_nbcross(self):
        np_result = np.cross(np.array([1, 2, 3]), np.array([4, 5, 6]))
        nb_result = nbcross(np.array([1, 2, 3]), np.array([4, 5, 6]))
        np.testing.assert_array_equal(nb_result, np_result)

        with self.assertRaises(ValueError):
            nb_result = nbcross(np.array([1]), np.array([4, 5, 6]))

    def test_nbdiag(self):
        np_result = np.diag(np.array([1, 2, 3]))
        nb_result = nbdiag(np.array([1, 2, 3]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbdiagflat(self):
        np_result = np.diagflat(np.array([1, 2, 3]))
        nb_result = nbdiagflat(np.array([1, 2, 3]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbdiff(self):
        np_result = np.diff(np.array([1, 2, 4, 7]))
        nb_result = nbdiff(np.array([1, 2, 4, 7]))
        np.testing.assert_array_equal(nb_result, np_result)

        np_result = np.diff(np.array([1, 2, 4, 7]), n=2)
        nb_result = nbdiff(np.array([1, 2, 4, 7]), n=2)
        np.testing.assert_array_equal(nb_result, np_result)

        with self.assertRaises(AssertionError):
            nb_result = nbdiff(np.array([[1, 2], [3, 4]]))

    def test_nbdigitize(self):
        np_result = np.digitize(np.array([1, 2, 3, 4]), bins=np.array([1, 2, 3]))
        nb_result = nbdigitize(np.array([1, 2, 3, 4]), bins=np.array([1, 2, 3]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbdstack(self):
        np_result = np.dstack((np.array([1, 2]), np.array([3, 4])))
        nb_result = nbdstack((np.array([1, 2]), np.array([3, 4])))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbediff1d(self):
        np_result = np.ediff1d(np.array([1, 2, 4, 7]))
        nb_result = nbediff1d(np.array([1, 2, 4, 7]))
        np.testing.assert_array_equal(nb_result, np_result)

        with self.assertRaises(ValueError):
            nb_result = nbediff1d(np.array([[1, 2], [3, 4]]))

    def test_nbexpand_dims(self):
        np_result = np.expand_dims(np.array([1, 2, 3]), axis=0)
        nb_result = nbexpand_dims(np.array([1, 2, 3]), axis=0)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbextract(self):
        np_result = np.extract(np.array([True, False, True]), np.array([1, 2, 3]))
        nb_result = nbextract(np.array([True, False, True]), np.array([1, 2, 3]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbeye(self):
        np_result = np.eye(3)
        nb_result = nbeye(3)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbfill_diagonal(self):
        a = np.zeros((3, 3), int)
        np.fill_diagonal(a, 5)
        nb_result = np.zeros((3, 3), int)
        nbfill_diagonal(nb_result, 5)
        np.testing.assert_array_equal(nb_result, a)

    def test_nbflatten(self):
        np_result = np.array([[1, 2], [3, 4]]).flatten()
        nb_result = nbflatten(np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbflatnonzero(self):
        np_result = np.flatnonzero(np.array([1, 0, 2, 0, 3, 0]))
        nb_result = nbflatnonzero(np.array([1, 0, 2, 0, 3, 0]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbflip(self):
        np_result = np.flip(np.array([1, 2, 3]))
        nb_result = nbflip(np.array([1, 2, 3]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbflipud(self):
        np_result = np.flipud(np.array([[1, 2], [3, 4], [5, 6]]))
        nb_result = nbflipud(np.array([[1, 2], [3, 4], [5, 6]]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbfliplr(self):
        np_result = np.fliplr(np.array([[1, 2], [3, 4], [5, 6]]))
        nb_result = nbfliplr(np.array([[1, 2], [3, 4], [5, 6]]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbfull(self):
        np_result = np.full(5, 3.14)
        nb_result = nbfull(5, 3.14)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbfull_like(self):
        np_result = np.full_like(np.array([1, 2, 3]), 3.14)
        nb_result = nbfull_like(np.array([1, 2, 3]), 3.14)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbgeomspace(self):
        np_result = np.geomspace(1.0, 1000.0, 4)
        nb_result = nbgeomspace(1.0, 1000.0, 4)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbhistogram(self):
        np_result = np.histogram(np.array([1, 2, 1, 2, 3, 1]))
        nb_result = nbhistogram(np.array([1, 2, 1, 2, 3, 1]))
        for nb_arr, np_arr in zip(nb_result, np_result):
            np.testing.assert_array_equal(nb_arr, np_arr)

    def test_nbhsplit(self):
        np_result = np.hsplit(np.array([[1, 2, 3], [4, 5, 6]]), 3)
        nb_result = nbhsplit(np.array([[1, 2, 3], [4, 5, 6]]), 3)
        for nb_arr, np_arr in zip(nb_result, np_result):
            np.testing.assert_array_equal(nb_arr, np_arr)

    def test_nbhstack(self):
        np_result = np.hstack((np.array([1, 2]), np.array([3, 4])))
        nb_result = nbhstack((np.array([1, 2]), np.array([3, 4])))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbidentity(self):
        np_result = np.identity(3)
        nb_result = nbidentity(3)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbindices(self):
        np_result = np.indices((2, 3))
        nb_result = nbindices((2, 3))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbinterp(self):
        np_result = np.interp(2.5, np.array([1, 2, 3]), np.array([1, 4, 9]))
        nb_result = nbinterp(2.5, np.array([1, 2, 3]), np.array([1, 4, 9]))
        self.assertEqual(nb_result, np_result)

    def test_nbintersect1d(self):
        np_result = np.intersect1d(np.array([1, 2, 3]), np.array([3, 4, 5]))
        nb_result = nbintersect1d(np.array([1, 2, 3]), np.array([3, 4, 5]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbisclose(self):
        np_result = np.isclose(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.1, 3.0]), rtol=1e-3
        )
        nb_result = nbisclose(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.1, 3.0]), rtol=1e-3
        )
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbiscomplex(self):
        np_result = np.iscomplex(np.array([1, 2, 3 + 1j]))
        nb_result = nbiscomplex(np.array([1, 2, 3 + 1j]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbiscomplexobj(self):
        np_result = np.iscomplexobj(np.array([1, 2, 3 + 1j]))
        nb_result = nbiscomplexobj(np.array([1, 2, 3 + 1j]))
        self.assertEqual(nb_result, np_result)

    def test_nbisneginf(self):
        np_result = np.isneginf(np.array([1, 2, -np.inf]))
        nb_result = nbisneginf(np.array([1, 2, -np.inf]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbisposinf(self):
        np_result = np.isposinf(np.array([1, 2, np.inf]))
        nb_result = nbisposinf(np.array([1, 2, np.inf]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbisreal(self):
        np_result = np.isreal(np.array([1, 2, 3 + 1j]))
        nb_result = nbisreal(np.array([1, 2, 3 + 1j]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbisrealobj(self):
        np_result = np.isrealobj(np.array([1, 2, 3 + 1j]))
        nb_result = nbisrealobj(np.array([1, 2, 3 + 1j]))
        self.assertEqual(nb_result, np_result)

    def test_nbishscalar(self):
        np_result = np.isscalar(3)
        nb_result = nbisscalar(3)
        self.assertEqual(nb_result, np_result)

    def test_nbkaiser(self):
        np_result = np.kaiser(5, 14)
        nb_result = nbkaiser(5, 14)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbnan_to_num(self):
        np_result = np.nan_to_num(np.array([1.0, 2.0, np.nan]))
        nb_result = nbnan_to_num(np.array([1.0, 2.0, np.nan]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbones(self):
        np_result = np.ones(5)
        nb_result = nbones(5)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbpartition(self):
        np_result = np.partition(np.array([3, 4, 2, 1]), 2)
        nb_result = nbpartition(np.array([3, 4, 2, 1]), 2)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbptp(self):
        np_result = np.ptp(np.array([1, 2, 3]))
        nb_result = nbptp(np.array([1, 2, 3]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbrepeat(self):
        np_result = np.repeat(np.array([1, 2, 3]), 2)
        nb_result = nbrepeat(np.array([1, 2, 3]), 2)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbreshape(self):
        np_result = np.reshape(np.array([1, 2, 3, 4]), (2, 2))
        nb_result = nbreshape(np.array([1, 2, 3, 4]), (2, 2))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbroll(self):
        np_result = np.roll(np.array([1, 2, 3]), 1, 0)
        nb_result = nbroll(np.array([1, 2, 3]), 1, 0)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbrot90(self):
        np_result = np.rot90(np.array([[1, 2], [3, 4]]))
        nb_result = nbrot90(np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbravel(self):
        np_result = np.ravel(np.array([[1, 2], [3, 4]]))
        nb_result = nbravel(np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbrow_stack(self):
        np_result = np.row_stack((np.array([1, 2]), np.array([3, 4])))
        nb_result = nbrow_stack((np.array([1, 2]), np.array([3, 4])))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbround(self):
        np_result = np.round(np.array([1.1234, 2.5678]), decimals=2)
        nb_result = nbround(np.array([1.1234, 2.5678]), decimals=2)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbsearchsorted(self):
        np_result = np.searchsorted(np.array([1, 2, 3, 4]), 3)
        nb_result = nbsearchsorted(np.array([1, 2, 3, 4]), 3)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbselect(self):
        np_result = np.select(
            [np.array([True, False]), np.array([False, True])],
            [np.array([1, 2]), np.array([3, 4])],
            default=0,
        )
        nb_result = nbselect(
            [np.array([True, False]), np.array([False, True])],
            [np.array([1, 2]), np.array([3, 4])],
            default=0,
        )
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbshape(self):
        np_result = np.shape(np.array([1, 2, 3]))
        nb_result = nbshape(np.array([1, 2, 3]))
        self.assertEqual(nb_result, np_result)

    def test_nbsort(self):
        np_result = np.sort(np.array([3, 1, 2]))
        nb_result = nbsort(np.array([3, 1, 2]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbsplit(self):
        np_result = np.split(np.array([1, 2, 3, 4, 5]), [2])
        nb_result = nbsplit(np.array([1, 2, 3, 4, 5]), [2])
        for nb_array, np_array in zip(nb_result, np_result):
            np.testing.assert_array_equal(nb_array, np_array)

    def test_nbstack(self):
        np_result = np.stack((np.array([1, 2]), np.array([3, 4])), axis=0)
        nb_result = nbstack((np.array([1, 2]), np.array([3, 4])), axis=0)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbswapaxes(self):
        np_result = np.swapaxes(np.array([[1, 2], [3, 4]]), 0, 1)
        nb_result = nbswapaxes(np.array([[1, 2], [3, 4]]), 0, 1)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbtake(self):
        np_result = np.take(np.array([1, 2, 3, 4]), [0, 2])
        nb_result = nbtake(np.array([1, 2, 3, 4]), [0, 2])
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbtranspose(self):
        np_result = np.transpose(np.array([[1, 2], [3, 4]]))
        nb_result = nbtranspose(np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbtri(self):
        np_result = np.tri(3, 3, k=-1)
        nb_result = nbtri(3, 3, k=-1)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbtril(self):
        np_result = np.tril(np.array([[1, 2], [3, 4]]), k=-1)
        nb_result = nbtril(np.array([[1, 2], [3, 4]]), k=-1)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbtril_indices(self):
        np_result = np.tril_indices(3, k=-1)
        nb_result = nbtril_indices(3, k=-1)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbtril_indices_from(self):
        np_result = np.tril_indices_from(np.array([[1, 2], [3, 4]]), k=-1)
        nb_result = nbtril_indices_from(np.array([[1, 2], [3, 4]]), k=-1)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbtriu(self):
        np_result = np.triu(np.array([[1, 2], [3, 4]]), k=-1)
        nb_result = nbtriu(np.array([[1, 2], [3, 4]]), k=-1)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbtriu_indices(self):
        np_result = np.triu_indices(3, k=-1)
        nb_result = nbtriu_indices(3, k=-1)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbtriu_indices_from(self):
        np_result = np.triu_indices_from(np.array([[1, 2], [3, 4]]), k=-1)
        nb_result = nbtriu_indices_from(np.array([[1, 2], [3, 4]]), k=-1)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbtrim_zeros(self):
        np_result = np.trim_zeros(np.array([0, 1, 2, 0, 3, 0]), "f")
        nb_result = nbtrim_zeros(np.array([0, 1, 2, 0, 3, 0]), "f")
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbunion1d(self):
        np_result = np.union1d(np.array([1, 2, 3]), np.array([3, 4, 5]))
        nb_result = nbunion1d(np.array([1, 2, 3]), np.array([3, 4, 5]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbunique(self):
        np_result = np.unique(np.array([1, 2, 2, 3, 3, 3]))
        nb_result = nbunique(np.array([1, 2, 2, 3, 3, 3]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbunwrap(self):
        np_result = np.unwrap(np.array([0, 2 * np.pi, 4 * np.pi]))
        nb_result = nbunwrap(np.array([0, 2 * np.pi, 4 * np.pi]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbvander(self):
        np_result = np.vander(np.array([1, 2, 3]), 3)
        nb_result = nbvander(np.array([1, 2, 3]), 3)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbvsplit(self):
        np_result = np.vsplit(np.array([[1, 2, 3], [4, 5, 6]]), 2)
        nb_result = nbvsplit(np.array([[1, 2, 3], [4, 5, 6]]), 2)
        for nb_array, np_array in zip(nb_result, np_result):
            np.testing.assert_array_equal(nb_array, np_array)

    def test_nbvstack(self):
        np_result = np.vstack((np.array([1, 2]), np.array([3, 4])))
        nb_result = nbvstack((np.array([1, 2]), np.array([3, 4])))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbwhere(self):
        np_result = np.where(
            np.array([True, False, True]), np.array([1, 2, 3]), np.array([4, 5, 6])
        )
        nb_result = nbwhere(
            np.array([True, False, True]), np.array([1, 2, 3]), np.array([4, 5, 6])
        )
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbzeros(self):
        np_result = np.zeros(5)
        nb_result = nbzeros(5)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbzeros_like(self):
        np_result = np.zeros_like(np.array([1, 2, 3]))
        nb_result = nbzeros_like(np.array([1, 2, 3]))
        np.testing.assert_array_equal(nb_result, np_result)


if __name__ == "__main__":
    unittest.main()
