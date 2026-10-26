import numpy as np
import scipy
from scipy.special import expit
import scipy.sparse  


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Parameters
    ----------
        A : np.array or scipy.sparse.csr_matrix of shape (m, n)
            Feature matrix (training sample).
        b : np.array of shape (m,)
            Labels vector with values in {-1, 1}.
        regcoef : float
            Regularization coefficient.
    """
    def __init__(self, A, b, regcoef):
        self.A = A
        self.b = b
        self.regcoef = regcoef
        self._m = A.shape[0]

    def func(self, x):
        # TODO: Implement

        z = self.b * self.A.dot(x)
        return float(np.sum(np.logaddexp(0.0, -z)) / self._m + self.regcoef / 2.0 * np.dot(x, x))

    def grad(self, x):
        # TODO: Implement

        z = self.b * self.A.dot(x)
        s = expit(-z)
        return -self.A.T.dot(self.b * s) / self._m + self.regcoef * x

    def hess(self, x):
        # TODO: Implement

        z = self.b * self.A.dot(x)
        sig = expit(-z)
        s = sig * (1.0 - sig)

        if scipy.sparse.issparse(self.A):
            A_scaled = self.A.multiply(s[:, None])
            H = A_scaled.T.dot(self.A)
            H = H.toarray()
        else:
            A_scaled = self.A * s[:, None]
            H = A_scaled.T.dot(self.A)

        return H / self._m + self.regcoef * np.eye(self.A.shape[1])


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the gradient

    x = np.asarray(x, dtype=float)
    n = x.size
    result = np.zeros(n)
    fx = func(x)

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        result[i] = (func(x + eps * e_i) - fx) / eps

    return result


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the Hessian

    x = np.asarray(x, dtype=float)
    n = x.size
    result = np.zeros((n, n))
    fx = func(x)

    E = np.eye(n) * eps
    f_plus = np.zeros(n)
    for i in range(n):
        f_plus[i] = func(x + E[i])

    for i in range(n):
        for j in range(n):
            result[i, j] = (
                func(x + E[i] + E[j])
                - f_plus[i]
                - f_plus[j]
                + fx
            ) / (eps ** 2)

    return result