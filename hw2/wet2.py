import numpy as np
import matplotlib.pyplot as plt
from mcholmz import modifiedChol as mchol


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
        grad[-1] =
        grad[0] = -2 + 2 * x[0]





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


def gradient_decent(f, x0, line_search=inexact_line_search, alpha=1, beta=0.5, max_iter=100000):
    """
    Gradient Descent with Inexact Line Search
    """
    x = x0
    for i in range(max_iter):
        grad = f.gradient(x)
        if np.linalg.norm(grad) < eps:
            break

        p = - grad / np.linalg.norm(grad)
        alpha = line_search(f, x, p, grad, alpha, beta)
        x = x + alpha * p
    return x


def newton_method(f, x0, line_search=inexact_line_search, alpha=1, beta=0.5, max_iter=100000):
    x = x0

    for i in range(max_iter):
        grad = f.gradient(x)
        hess = f.hessian(x)

        L, d, _ = mchol(hess)

        # we now solve LDL^T @ p = -grad
        # we define y = D @ L^-1 @ p
        y = np.linalg.solve(L, -grad)
        p = np.linalg.solve(np.diag(d.flatten()) @ L.T, y)

        lambda_squared = - grad.T @ p
        if lambda_squared / 2 < eps:
            break

        p = p / np.linalg.norm(p)

        alpha = line_search(f, x, p, grad, alpha, beta)
        x = x + alpha * p

    return x


if __name__ == "__main__":
    f1 = f_quadratic(np.array([[3, 0],
                               [0, 3]]))
    f2_3 = f_quadratic(np.array([[10, 0],
                                [0, 1]]))
    x0 = np.array([[1.5],
                   [2]])

    # x1_gd = gradient_decent(f1, x0, line_search=exact_line_search)
    # print(x1_gd)
    x1_nt = newton_method(f1, x0, line_search=exact_line_search)
    print(x1_nt)

    # x2_gd = gradient_decent(f2_3, x0, line_search=exact_line_search)
    # print(x2_gd)
    x2_nt = newton_method(f2_3, x0, line_search=exact_line_search)
    print(x2_nt)

    # x3_gd = gradient_decent(f2_3, x0, line_search=inexact_line_search)
    # print(x3_gd)
    x3_nt = newton_method(f2_3, x0, line_search=inexact_line_search)
    print(x3_nt)


