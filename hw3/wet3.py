import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append('..')

#### parameters
eps = 1e-5
c_1 = 0.25
c_2 = 0.5

class NeuralNetwork:
    def __init__(self, d_in=2):
        self.d_in = d_in
        self.dims = [2, 4, 3, 1]
        # construct the Weight matrices
        self.W = [ np.random.rand(din,dout) for din,dout in zip(self.dims[:-1],self.dims[1:])]

    def __call__(self, x):
        assert x.shape[0] == self.d_in, "incompatible input"


class f_xexp:
    # Q1.3.5
    def __call__(self, x):
        x = x.reshape(-1,2)
        x1, x2 = x[:,0], x[:,1]
        return x1 * np.exp(-1*(x1**2 + x2**2))

class activation_fn:
    # Q 1.3.6
    def __call__(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def grad(self, x):
        return 1 - activation_fn(x) ** 2

class loss_fn:
    # Q 1.3.7
    def __call__(self, y, y_hat):
        assert y.shape == y_hat.shape
        return np.sum( np.linalg.norm(y - y_hat) ** 2 ) / y.shape[0]

    def grad(self,y , y_hat):
        assert y.shape == y_hat.shape
        return 2 * np.sum(y - y_hat) / y.shape[0]


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
        y, s = y.reshape(-1,1), s.reshape(-1,1)
        H_inv = (np.eye(x.shape[0]) - rho * y @ s.T) @ H_inv @ \
                (np.eye(x.shape[0]) - rho * s @ y.T) + rho * s @ s.T
        p = - H_inv @ grad_new

        # update
        x = x_new
        grad = grad_new

        # print
        # print('iter: {}, x: {}, f: {}'.format(i, x, f(x)))

    return x, x_vals, f_vals


def generate_dataset(n):
    """
    samples dataset set from function
    :param n: dataset size
    :return:
    """
    # default rand samples on [0,1) so extending to [-2,2)
    x_ds = random.random.rand(n,2) * 4 - 2
    fn = f_xexp()
    y_ds = fn(x_ds)
    return x_ds, y_ds


if __name__ == '__main__':
    # Rosenbrock
    f = FRosenbrock(n=10)
    x = np.zeros([10])
    x_bfgs, x_vals, f_vals = bfgs(f, x)
    print('x_bfgs: {}'.format(x_bfgs))

    # plot
    x_vals = np.array(x_vals)
    f_vals = np.array(f_vals)

    # Q1.2 plot log scale graph of result f_vals
    ax = plt.axes()
    ax.set_ylabel("log")
    plt.xlabel("iteration")
    ax.set_title("Distance of Rosebrock function value from optimum -logscale")
    ax.plot(f_vals)
    # plt.show()

    # Q1.3.10
    x_ds, y_ds = generate_dataset(n=500)

