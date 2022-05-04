import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from hw2.mcholmz import modifiedChol as mchol


#### parameters
eps = 1e-5
sigma = 0.25


class f_quadratic:
    def __init__(self, Q):
        self.Q = Q

    def __call__(self, x):
        return 0.5 * x.T @ self.Q @ x

    def gradient(self, x):
        return 0.5 * (self.Q + self.Q.T) @ x

    def hessian(self, x):
        return 0.5 * (self.Q + self.Q.T)


class f_rosenbrock:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def gradient(self, x):
        grad = np.zeros(self.n)
        grad[0] = -2 + 2 * x[0] + 400 * x[0] * (- x[1] + x[0]**2)
        grad[-1] = 200 * (x[-1] - x[-2]**2)
        for i in range(1, self.n - 1):
            grad[i] = -2 + 202 * x[i] -400 * x[i] * x[i+1] + 400 * x[i]**3 -200 * x[i-1]**2

        return grad

    def hessian(self, x):
        hess = np.zeros((self.n, self.n))
        hess[0, 0] = 2 + 1200 * x[0]**2 - 400 * x[1] + 2 * x[0]
        hess[-1, -1] = 200
        hess[0, 1] = hess[1, 0] = -400 * x[0]
        hess[-1, -2] = hess[-2, -1] = -400 * x[-2]
        for i in range(1, self.n - 1):
            hess[i, i] = 202 + 1200 * x[i]**2 - 400 * x[i+1]
            hess[i-1, i] = hess[i, i-1] = -400 * x[i]

        return hess


def inexact_line_search(f, x, p, grad, alpha, beta):
    """
    Inexact Line Search
    from wikipedia:
    sigma is c
    m is grad^T @ p
    """
    while f(x) - f(x + alpha * p) < - alpha * sigma * p.T @ grad:
        alpha = beta * alpha

    return alpha


def exact_line_search(f, x, p, grad, alpha, beta):
    """
    assumes that f is quadratic.
    grad, alpha and beta arguments are not used,
    they are there for consistency with inexact_line_search
    """
    assert isinstance(f, f_quadratic)
    return - (x.T @ (f.Q + f.Q.T) @ p / (2 * p.T @ f.Q @ p))


def gradient_decent(f, x0, line_search=inexact_line_search, alpha=1, beta=0.5, max_iter=1000000):
    """
    Gradient Descent with Inexact Line Search
    """
    x_vals = []
    f_vals = []

    x = x0
    for i in range(max_iter):
        x_vals.append(x)
        f_vals.append(f(x))

        grad = f.gradient(x)
        if np.linalg.norm(grad) < eps:
            print("gradient descent converged")
            break

        p = - grad / np.linalg.norm(grad)
        alpha = line_search(f, x, p, grad, alpha, beta)
        x = x + alpha * p
    return x, x_vals, f_vals


def newton_method(f, x0, line_search=inexact_line_search, alpha=1, beta=0.5, max_iter=100000):
    x_vals = []
    f_vals = []

    x = x0
    for i in range(max_iter):
        x_vals.append(x)
        f_vals.append(f(x))

        grad = f.gradient(x)
        hess = f.hessian(x)

        L, d, _ = mchol(hess)

        # we now solve LDL^T @ p = -grad
        # we define y = D @ L^-1 @ p
        y = np.linalg.solve(L, -grad)
        p = np.linalg.solve(np.diag(d.flatten()) @ L.T, y)

        lambda_squared = - grad.T @ p
        if np.linalg.norm(grad) < eps:  # lambda_squared < eps:
            print("newton method converged")
            break

        p = p / np.linalg.norm(p)

        alpha = line_search(f, x, p, grad, alpha, beta)
        x = x + alpha * p

    return x, x_vals, f_vals


if __name__ == "__main__":

    # section 2.5

    # f1 = f_quadratic(np.array([[3, 0],
    #                            [0, 3]]))
    # f2_3 = f_quadratic(np.array([[10, 0],
    #                             [0, 1]]))
    # x0 = np.array([[1.5],
    #                [2]])

    # x1_gd = gradient_decent(f1, x0, line_search=exact_line_search)
    # print(x1_gd)
    # x1_nt = newton_method(f1, x0, line_search=exact_line_search)
    # print(x1_nt)

    # x2_gd = gradient_decent(f2_3, x0, line_search=exact_line_search)
    # print(x2_gd)
    # x2_nt = newton_method(f2_3, x0, line_search=exact_line_search)
    # print(x2_nt)

    # x3_gd = gradient_decent(f2_3, x0, line_search=inexact_line_search)
    # print(x3_gd)
    # x3_nt = newton_method(f2_3, x0, line_search=inexact_line_search)
    # print(x3_nt)

    # section 2.6
    f_rosenbrock = f_rosenbrock(10)
    x0 = np.zeros(10)
    _, _, f_vals_grad = gradient_decent(f_rosenbrock, x0)
    _, _, f_vals_nt = newton_method(f_rosenbrock, x0)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(f_vals_grad)
    axs[0].set_title("10D Rosenbrock With Gradient Descent")
    axs[1].plot(f_vals_nt)
    axs[1].set_title("10D Rosenbrock With Newton Method")
    for ax in axs.flat:
        ax.set(xlabel='iteration', ylabel='error')
        ax.set_yscale('log')
    fig.tight_layout()
    plt.show()



