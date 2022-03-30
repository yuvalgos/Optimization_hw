import numpy as np
import matplotlib.pyplot as plt


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


def numerical_grad(f, x, eps=1e-5, *args):
    grad = (f(x+eps, *args) - f(x-eps, *args)) / (2*eps)
    return grad


def numerical_hessian(f, x, eps=1e-5, *args):
    hessian = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        hessian[i, :] = numerical_grad(f, x + eps, eps, *args) - numerical_grad(f, x - eps, eps, *args)
        hessian[i, :] /= 2*eps

    return hessian


def f1(x, A):
    return phi113(A @ x)


def f2(x):
    return h(phi113(x))


if __name__ == '__main__':
    x = np.random.randn(3)/5
    A = np.random.randn(3, 3)/5

    epsilons = [2**-i for i in range(60)]

    _, f1_grad_analytic, f1_hess_analytic = sec1151(x, A)
    _, f2_grad_analytic, f2_hess_analytic = sec1152(x)

    err_grad1, err_hess1 = [], []
    err_grad2, err_hess2 = [], []
    for e in epsilons:
        f1_grad_numerical = numerical_grad(f1, x, e, A)
        err_grad1.append(np.linalg.norm(f1_grad_numerical - f1_grad_analytic, ord=np.inf))
        f1_hess_numerical = numerical_hessian(f1, x, e, A)
        err_hess1.append(np.linalg.norm((f1_hess_numerical - f1_hess_analytic).reshape(-1), ord=np.inf))
        f2_grad_numerical = numerical_grad(f2, x, e)
        err_grad2.append(np.linalg.norm(f2_grad_numerical - f2_grad_analytic, ord=np.inf))
        f2_hess_numerical = numerical_hessian(f2, x, e)
        err_hess2.append(np.linalg.norm((f2_hess_numerical - f2_hess_analytic).reshape(-1), ord=np.inf))

    epsilons_axis = list(range(60))

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(epsilons_axis, err_grad1)
    axs[0, 0].set_title('Gradient f1 Error')
    axs[0, 1].plot(epsilons_axis, err_hess1)
    axs[0, 1].set_title('Hessian f1 Error')
    axs[1, 0].plot(epsilons_axis, err_grad2)
    axs[1, 0].set_title('Gradient f2 Error')
    axs[1, 1].plot(epsilons_axis, err_hess2)
    axs[1, 1].set_title('Hessian f2 Error')

    for ax in axs.flat:
        ax.set(xlabel='-log(epsilon)', ylabel='error')
        ax.set_yscale('log')
    fig.tight_layout()
    plt.show()

    print(A)
    print(x)