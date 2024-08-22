import unittest
import numpy as np
from mm_toolbox.src.numba import (
    nblinspace,
    nbgeomspace,
    nbarange,
    nblogspace,
    nbzeros,
    nbones,
    nbfull,
    nbeye,
    nbdiag,
    nbisin,
    nbwhere,
    nbdiff,
    nbflip,
    nbsort,
    nbargsort,
    nbconcatenate,
    nbravel,
    nbreshape,
    nbtranspose,
    nbhstack,
    nbvstack,
    nbclip,
    nbunique,
    nbrepeat,
    nbstack,
    nbroll
)

class TestNumbaFuncs(unittest.TestCase):
    def test_nblinspace(self):
        np_result = np.linspace(0.0, 100.0, 10)
        nb_result = nblinspace(0.0, 100.0, 10)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbgeomspace(self):
        np_result = np.geomspace(1.0, 1000.0, 4)
        nb_result = nbgeomspace(1.0, 1000.0, 4)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbarange(self):
        np_result = np.arange(0.0, 10.0, 2.0)
        nb_result = nbarange(0.0, 10.0, 2.0)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nblogspace(self):
        np_result = np.logspace(1.0, 1000.0, 4)
        nb_result = nblogspace(1.0, 1000.0, 4)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbzeros(self):
        np_result = np.zeros(5)
        nb_result = nbzeros(5)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbones(self):
        np_result = np.ones(5)
        nb_result = nbones(5)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbfull(self):
        np_result = np.full(5, 3.14)
        nb_result = nbfull(5, 3.14)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbeye(self):
        np_result = np.eye(3)
        nb_result = nbeye(3)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbdiag(self):
        np_result = np.diag(np.array([1, 2, 3]))
        nb_result = nbdiag(np.array([1, 2, 3]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbisin(self):
        np_result = np.isin(np.array([1, 2, 3]), np.array([2, 3, 4]))
        nb_result = nbisin(np.array([1, 2, 3]), np.array([2, 3, 4]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbwhere(self):
        np_result = np.where(np.array([True, False, True]), np.array([1, 2, 3]), np.array([4, 5, 6]))
        nb_result = nbwhere(np.array([True, False, True]), np.array([1, 2, 3]), np.array([4, 5, 6]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbdiff(self):
        np_result = np.diff(np.array([1, 2, 4, 7]))
        nb_result = nbdiff(np.array([1, 2, 4, 7]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbflip(self):
        np_result = np.flip(np.array([1, 2, 3]))
        nb_result = nbflip(np.array([1, 2, 3]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbsort(self):
        np_result = np.sort(np.array([3, 1, 2]))
        nb_result = nbsort(np.array([3, 1, 2]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbargsort(self):
        np_result = np.argsort(np.array([3, 1, 2]))
        nb_result = nbargsort(np.array([3, 1, 2]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbconcatenate(self):
        np_result = np.concatenate((np.array([1, 2]), np.array([3, 4])))
        nb_result = nbconcatenate((np.array([1, 2]), np.array([3, 4])))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbravel(self):
        np_result = np.ravel(np.array([[1, 2], [3, 4]]))
        nb_result = nbravel(np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbreshape(self):
        np_result = np.reshape(np.array([1, 2, 3, 4]), (2, 2))
        nb_result = nbreshape(np.array([1, 2, 3, 4]), (2, 2))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbtranspose(self):
        np_result = np.transpose(np.array([[1, 2], [3, 4]]))
        nb_result = nbtranspose(np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbhstack(self):
        np_result = np.hstack((np.array([1, 2]), np.array([3, 4])))
        nb_result = nbhstack((np.array([1, 2]), np.array([3, 4])))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbvstack(self):
        np_result = np.vstack((np.array([1, 2]), np.array([3, 4])))
        nb_result = nbvstack((np.array([1, 2]), np.array([3, 4])))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbclip(self):
        np_result = np.clip(np.array([1, 2, 3, 4, 5]), 2, 4)
        nb_result = nbclip(np.array([1, 2, 3, 4, 5]), 2, 4)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbunique(self):
        np_result = np.unique(np.array([1, 2, 2, 3, 3, 3]))
        nb_result = nbunique(np.array([1, 2, 2, 3, 3, 3]))
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbrepeat(self):
        np_result = np.repeat(np.array([1, 2, 3]), 2)
        nb_result = nbrepeat(np.array([1, 2, 3]), 2)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbstack(self):
        np_result = np.stack((np.array([1, 2]), np.array([3, 4])), axis=0)
        nb_result = nbstack((np.array([1, 2]), np.array([3, 4])), axis=0)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbroll(self):
        np_result = np.roll(np.array([1, 2, 3]), 1, 0)
        nb_result = nbroll(np.array([1, 2, 3]), 1, 0)
        np.testing.assert_array_equal(nb_result, np_result)

if __name__ == "__main__":
    unittest.main()
