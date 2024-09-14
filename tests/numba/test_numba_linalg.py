import unittest
import numpy as np
from mm_toolbox.numba.linalg import (
    nbcholesky as nbcholesky,
    nbcond as nbcond,
    nbcov as nbcov,
    nbdet as nbdet,
    nbdot as nbdot,
    nbeig as nbeig,
    nbeigh as nbeigh,
    nbeigvals as nbeigvals,
    nbeigvalsh as nbeigvalsh,
    nbinv as nbinv,
    nbkron as nbkron,
    nblstsq as nblstsq,
    nbmatrix_power as nbmatrix_power,
    nbmatrix_rank as nbmatrix_rank,
    nbnorm as nbnorm,
    nbouter as nbouter,
    nbpinv as nbpinv,
    nbqr as nbqr,
    nbslodet as nbslodet,
    nbsolve as nbsolve,
    nbsvd as nbsvd,
    nbtrace as nbtrace,
    nbvdot as nbvdot,
)


class TestNumbaLinalgFuncs(unittest.TestCase):
    def test_nbcov(self):
        np_result = np.cov(np.array([[0.3, 2.9, 1.5], [1.7, 1.8, 0.5]]))
        nb_result = nbcov(np.array([[0.3, 2.9, 1.5], [1.7, 1.8, 0.5]]))
        np.testing.assert_array_equal(nb_result, np_result)

        with self.assertRaises(ValueError):
            nb_result = nbcov(np.array([1, 2, 3]), np.array([1, 2]))

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
        A = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        B = np.array([9.0, 8.0])
        np_result = np.linalg.solve(A, B)
        nb_result = nbsolve(A, B)
        np.testing.assert_array_almost_equal(nb_result, np_result)

    def test_nbinv(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        np_result = np.linalg.inv(A)
        nb_result = nbinv(A)
        np.testing.assert_array_almost_equal(nb_result, np_result)

    def test_nbdet(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
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
        A = np.array([[1, -2j], [2j, 5]], dtype=np.complex128)
        np_result = np.linalg.cholesky(A)
        nb_result = nbcholesky(A)
        np.testing.assert_array_almost_equal(nb_result, np_result)

    def test_nbkron(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        B = np.array([[0, 5], [6, 7]], dtype=np.float64)
        np_result = np.kron(A, B)
        nb_result = nbkron(A, B)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbouter(self):
        A = np.array([1, 2, 3])
        B = np.array([0, 1, 0])
        np_result = np.outer(A, B)
        nb_result = nbouter(A, B)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbtrace(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        np_result = np.trace(A)
        nb_result = nbtrace(A)
        self.assertEqual(nb_result, np_result)

    def test_nbvdot(self):
        A = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        B = np.array([5 + 6j, 7 + 8j], dtype=np.complex128)
        np_result = np.vdot(A, B)
        nb_result = nbvdot(A, B)
        self.assertEqual(nb_result, np_result)

    def test_nbcond(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        np_result = np.linalg.cond(A)
        nb_result = nbcond(A)
        self.assertAlmostEqual(nb_result, np_result)

    def test_nbeigh(self):
        A = np.array([[2, -1], [-1, 2]], dtype=np.float64)
        np_eigvals, np_eigvecs = np.linalg.eigh(A)
        nb_eigvals, nb_eigvecs = nbeigh(A)
        np.testing.assert_array_almost_equal(nb_eigvals, np_eigvals)
        np.testing.assert_array_almost_equal(nb_eigvecs, np_eigvecs)

    def test_nbeigvals(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        np_result = np.linalg.eigvals(A)
        nb_result = nbeigvals(A)
        np.testing.assert_array_almost_equal(nb_result, np_result)

    def test_nbeigvalsh(self):
        A = np.array([[2, -1], [-1, 2]], dtype=np.float64)
        np_result = np.linalg.eigvalsh(A)
        nb_result = nbeigvalsh(A)
        np.testing.assert_array_almost_equal(nb_result, np_result)

    def test_nblstsq(self):
        A = np.array([[1, 1], [1, -1], [1, 1]], dtype=np.float64)
        B = np.array([1, 0, -1], dtype=np.float64)
        np_result = np.linalg.lstsq(A, B, rcond=None)
        nb_result = nblstsq(A, B)
        for np_array, nb_array in zip(np_result, nb_result):
            np.testing.assert_array_almost_equal(nb_array, np_array)

    def test_nbmatrix_power(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        np_result = np.linalg.matrix_power(A, 2)
        nb_result = nbmatrix_power(A, 2)
        np.testing.assert_array_equal(nb_result, np_result)

    def test_nbmatrix_rank(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        np_result = np.linalg.matrix_rank(A)
        nb_result = nbmatrix_rank(A)
        self.assertEqual(nb_result, np_result)

    def test_nbpinv(self):
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        np_result = np.linalg.pinv(A)
        nb_result = nbpinv(A)
        np.testing.assert_array_almost_equal(nb_result, np_result)

    def test_nbslodet(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        np_result = np.linalg.slogdet(A)
        nb_result = nbslodet(A)
        self.assertAlmostEqual(nb_result[0], np_result[0])
        self.assertAlmostEqual(nb_result[1], np_result[1])


if __name__ == "__main__":
    unittest.main()
