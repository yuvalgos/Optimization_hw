import numpy as np


def phi113(x):
    x1, x2, x3 = x
    return (np.cos(x1 * x2**2 *x3))**2


def phi113_grad(x):
    x1, x2, x3 = x
    return np.array([
        -x2**2 * x3 * np.sin(2 * x1 * x2**2 * x3),
        -2 * x1 * x2 * x3 * np.sin(2 * x1 * x2**2 * x3),
        -x2**2 *x1 * np.sin(2 * x1 * x2**2 * x3)
    ])


def phi113_hessian(x):
    x1, x2, x3 = x
    h11 = -2 * x2**4 * x3**2 * np.sin(2 * x1 * x2**2 * x3)
    h12 = h21 = -2 * x2 * x3 * np.sin(2 * x1 * x2**2 * x3) +\
        - 4 * x2**3 * x3**2 * x1 * np.cos(2 * x1 * x2**2 * x3)
    h13 = h31 = -x2**2 * np.sin(2 * x1 * x2**2 * x3) -\
        2 * x2**4 * x1 *x3 * np.cos(2 * x1 * x2**2 * x3)
    h22 = -2 * x1 * x3 * np.sin(2 * x1 * x2**2 * x3) -\
        8 * x1**2 * x2**2 * x3**2 * np.cos(2 * x1 * x2**2 * x3)
    h23 = h32 = -2 * x1 * x2 * np.sin(2 * x1 * x2**2 * x3) -\
        4 * x1**2 * x2**3 * x3 * np.cos(2 * x1 * x2**2 * x3)
    h33 = -2 * x1**2 * x2**4 * np.cos(2 * x1 * x2**2 * x3)

    return np.array([[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]])


def h(x):
    return np.sqrt(1 + np.sin(x)**2)


def h_tag(x):
    res = np.sin(2*x) / 2
    return res / np.sqrt(1 + np.sin(x)**2)


def h_tagtag(x):
    res = np.cos(2*x) / np.sqrt(1 + np.sin(x)**2)
    res += np.sin(2*x)**2 / (1 + np.sin(x)**2)**1.5
    return res


def sec1151(x: np.ndarray, A: np.ndarray):
    val = phi113(A @ x)
    grad = A.T @ phi113_grad(A@x)
    hessian = A.T @ phi113_hessian(A@x) @ A

    return val, grad, hessian


def sec1152(x: np.ndarray):
    val = h(phi113(x))
    grad = h_tag(phi113(x)) * phi113_grad(x)
    hessian = h_tagtag(phi113(x)) * phi113_grad(x) @ phi113_grad(x).T
    hessian += h_tag(phi113(x)) * phi113_hessian(x)

    return val, grad, hessian


def numerical_grad(f, x, eps=1e-5):
    grad = (f(x+eps) - f(x-eps)) / (2*eps)


def numerical_hessian(f, x, eps=1e-5):
    hessian = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        hessian[i, :] = numerical_grad(f, x + eps, eps) - numerical_grad(f, x - eps, eps)
        hessian[i, :] /= 2*eps

    return hessian


if __name__ == '__main__':
    x = np.random.randn(3)
    A = np.random.randn(3, 3)

    epsilons = [2**-i for i in range(1, 60)]

    f1_grad_a =

    for e in epsilons:



