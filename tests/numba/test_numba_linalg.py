import unittest
import numpy as np
from src.mm_toolbox.numba import (
    nbdot,
    nbnorm,
    nbsolve,
    nbinv,
    nbdet,
    nbeig,
    nbsvd,
    nbqr,
    nbcholesky,
)


class TestNumbaLinalgFuncs(unittest.TestCase):
    def test_nbdot(self):
        A = np.array([1.0, 2.0, 3.0])
        B = np.array([4.0, 5.0, 6.0])
        np_result = np.dot(A, B)
        nb_result = nbdot(A, B)
        self.assertAlmostEqual(nb_result, np_result)

    def test_nbnorm(self):
        A = np.array([1.0, 2.0, 3.0])
        np_result = np.linalg.norm(A)
        nb_result = nbnorm(A)
        self.assertAlmostEqual(nb_result, np_result)

    def test_nbsolve(self):
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        B = np.array([9.0, 8.0])
        np_result = np.linalg.solve(A, B)
        nb_result = nbsolve(A, B)
        np.testing.assert_array_almost_equal(nb_result, np_result)

    def test_nbinv(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        np_result = np.linalg.inv(A)
        nb_result = nbinv(A)
        np.testing.assert_array_almost_equal(nb_result, np_result)

    def test_nbdet(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        np_result = np.linalg.det(A)
        nb_result = nbdet(A)
        self.assertAlmostEqual(nb_result, np_result)

    def test_nbeig(self):
        A = np.array([[1.0, -1.0], [1.0, 1.0]], dtype=np.complex128)
        np_eigvals, np_eigvecs = np.linalg.eig(A)
        nb_eigvals, nb_eigvecs = nbeig(A)
        np.testing.assert_array_almost_equal(nb_eigvals, np_eigvals)
        np.testing.assert_array_almost_equal(nb_eigvecs, np_eigvecs)

    def test_nbsvd(self):
        A = np.random.randn(9, 6) + 1j * np.random.randn(9, 6)
        np_U, np_S, np_VT = np.linalg.svd(A, True)
        nb_U, nb_S, nb_VT = nbsvd(A, True)
        np.testing.assert_array_almost_equal(nb_U, np_U)
        np.testing.assert_array_almost_equal(nb_S, np_S)
        np.testing.assert_array_almost_equal(nb_VT, np_VT)

    def test_nbqr(self):
        A = np.random.randn(9, 6)
        np_Q, np_R = np.linalg.qr(A)
        nb_Q, nb_R = nbqr(A)
        np.testing.assert_array_almost_equal(nb_Q, np_Q)
        np.testing.assert_array_almost_equal(nb_R, np_R)

    def test_nbcholesky(self):
        A = np.array([[1, -2j], [2j, 5]])
        np_result = np.linalg.cholesky(A)
        nb_result = nbcholesky(A)
        np.testing.assert_array_almost_equal(nb_result, np_result)


if __name__ == "__main__":
    unittest.main()
