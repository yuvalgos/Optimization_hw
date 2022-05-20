import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append('..')

#### parameters
eps = 1e-5
c_1 = 0.25
c_2 = 0.5


class FRosenbrock:
    def __init__(self, n=10):
        self.n = n

    def __call__(self, x):
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    def gradient(self, x):
        grad = np.zeros(self.n)
        grad[0] = -2 + 2 * x[0] + 400 * x[0] * (- x[1] + x[0] ** 2)
        grad[-1] = 200 * (x[-1] - x[-2] ** 2)
        for i in range(1, self.n - 1):
            grad[i] = -2 + 202 * x[i] - 400 * x[i] * x[i + 1] + 400 * x[i] ** 3 - 200 * x[i - 1] ** 2

        return grad

    def hessian(self, x):
        hess = np.zeros((self.n, self.n))
        for i in range(0, self.n - 1):
            hess[i, i] = 202 + 1200 * x[i] ** 2 - 400 * x[i + 1]
            hess[i + 1, i] = hess[i, i + 1] = -400 * x[i]

        hess[0, 0] = 2 + 1200 * x[0] ** 2 - 400 * x[1]
        hess[-1, -1] = 200
        hess[0, 1] = hess[1, 0] = -400 * x[0]
        hess[-1, -2] = hess[-2, -1] = -400 * x[-2]

        return hess


def inexact_line_search(f, x, p, grad, alpha, beta):
    """
    Inexact Line Search
    from wikipedia:
    sigma is c
    m is grad^T @ p
    """
    while f(x) - f(x + alpha * p) < - alpha * c_1 * p.T @ grad \
            and f.gradient(x + alpha * p).T @ p > c_2 * grad.T @ p:
        alpha = beta * alpha

    return alpha


def bfgs(f, x, max_iter=10000):
    """
    BFGS
    """

    x_vals = []
    f_vals = []

    # initialize
    grad = f.gradient(x)
    H_inv = np.eye(x.shape[0])
    p = - H_inv @ grad
    alpha = 1.0

    # iteration
    for i in range(max_iter):
        f_vals.append(f(x))
        x_vals.append(x)

        if np.linalg.norm(f.gradient(x)) < eps:
            break

        # line search
        alpha = inexact_line_search(f, x, p, grad, alpha, 0.5)

        # update
        x_new = x + alpha * p
        grad_new = f.gradient(x_new)
        y = grad_new - grad
        s = x_new - x
        rho = 1 / (y.T @ s)
        H_inv = (np.eye(x.shape[0]) - rho * y @ s.T) @ H_inv @ \
                (np.eye(x.shape[0]) - rho * s @ y.T) + rho * s @ s.T
        p = - H_inv @ grad_new

        # update
        x = x_new
        grad = grad_new

        # print
        # print('iter: {}, x: {}, f: {}'.format(i, x, f(x)))

    return x, x_vals, f_vals


if __name__ == '__main__':
    # Rosenbrock
    f = FRosenbrock(n=10)
    x = np.zeros([10])
    x_bfgs, x_vals, f_vals = bfgs(f, x)
    print('x_bfgs: {}'.format(x_bfgs))

    # plot
    x_vals = np.array(x_vals)
    f_vals = np.array(f_vals)
    plt.plot(x_vals, f_vals)
    plt.show()