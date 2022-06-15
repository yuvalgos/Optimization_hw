import numpy as np
from mcholmz import modifiedChol as mchol


eps = 1e-5
sigma = 0.25


def inexact_line_search(f, x, lambda_, p, grad, alpha, beta):
    """
    Inexact Line Search
    from wikipedia:
    sigma is c
    m is grad^T @ p
    """
    while f(x, lambda_) - f(x + alpha * p, lambda_) < - alpha * sigma * p.T @ grad:
        alpha = beta * alpha

    return alpha


def newton_method(f, x0, lambda_, line_search=inexact_line_search, alpha=1, beta=0.5, max_iter=10000):
    x_vals = []
    f_vals = []

    x = x0
    for i in range(max_iter):
        x_vals.append(x)
        f_vals.append(f(x, lambda_))

        grad = f.grad(x, lambda_)
        hess = f.hessian(x, lambda_)

        L, d, _ = mchol(hess)

        # we now solve LDL^T @ p = -grad
        # we define y = D @ L^-1 @ p
        y = np.linalg.solve(L, -grad)
        p = np.linalg.solve(np.diag(d.flatten()) @ L.T, y)

        # lambda_squared = - grad.T @ p
        if np.linalg.norm(grad) < eps:  # lambda_squared < eps:
            print("newton method converged")
            break

        p = p / np.linalg.norm(p)

        alpha = line_search(f, x, lambda_, p, grad, alpha, beta)
        x = x + alpha * p

    return x, x_vals, f_vals