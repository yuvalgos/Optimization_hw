import numpy as np
from matplotlib import pyplot as plt

from newton_method import newton_method


class f:
    def __call__(self, x):
        return 2 * (x[0]-5)**2 + (x[1] - 1) ** 2

    def grad(self, x):
        return np.array([4*(x[0]-5), 2*(x[1]-1)])

    def hessian(self, x):
        return np.array([[4, 0],
                         [0, 2]])


class g1:
    def __call__(self, x):
        return 0.5 * x[0] + x[1] - 1

    def grad(self, x):
        return np.array([0.5, 1])

    def hessian(self, x):
        return np.zeros((2, 2))


class g2:
    def __call__(self, x):
        return x[0] - x[1]

    def grad(self, x):
        return np.array([1, -1])

    def hessian(self, x):
        return np.zeros((2, 2))


class g3:
    def __call__(self, x):
        return -x[0] - x[1]

    def grad(self, x):
        return np.array([-1, -1])

    def hessian(self, x):
        return np.zeros((2, 2))


class phi_p:
    def __call__(self, x, p):
        """ compute phi(x) elementwise if x is a vector"""
        return self._phi(p*x) / p

    def derivative(self, x, p):
        return self._derivative_phi(p*x)

    def _phi(self, x):
        res = np.where(x >= -0.5, 0.5 * x ** 2 + x, -0.25 * np.log(-2 * x) - 3/8)
        return res

    def _derivative_phi(self, x):
        return np.where(x >= -0.5, x + 1, -0.25/x)


class lagrangian:
    def __init__(self, f, constraint_funcs, phi_p, p):
        self.f = f
        self.constraints = constraint_funcs
        self.phi_p = phi_p
        self.p = p

    def __call__(self, x, lambda_):
        res = self.f(x)
        for i in range(len(self.constraints)):
            res += lambda_[i] * self.phi_p(self.constraints[i](x), self.p)
        return res

    def grad(self, x, lambda_):
        grad = self.f.grad(x)
        for i in range(len(self.constraints)):
            grad = grad + lambda_[i] *\
                    self.phi_p.derivative(self.constraints[i](x), self.p) *\
                    self.constraints[i].grad(x)

        return grad

    def hessian(self, x, lambda_):
        hessian = self.f.hessian(x)
        # all constraints we use are linear so their hessian are zeros and we can ignore them
        return hessian


def augmented_lagrangian(lagrangian, x0, lambda_0, p_max, alpha=2, max_iter=100):
    x = x0
    lambda_ = lambda_0
    plt_dict = dict(augmented_grads=[], f_vals=[], max_violations=[], x_list=[], lam_list=[], newton_iter=0)
    newton_iter = 0

    for i in range(max_iter):
        new_x, x_vals, lag_vals, grad_vals = newton_method(lagrangian, x, lambda_)
        newton_iter += len(x_vals)

        new_lambda = np.zeros(len(lambda_))
        for j in range(len(lambda_)):
            new_lambda[j] = lambda_[j] * lagrangian.phi_p.derivative(lagrangian.constraints[j](new_x), lagrangian.p)

        lagrangian.p = min(p_max, alpha * lagrangian.p)

        x = new_x
        lambda_ = new_lambda

        plt_dict['newton_iter'] = newton_iter
        plt_dict['x_list'].extend(x_vals)
        plt_dict['lam_list'].extend([lambda_] * len(x_vals))
        plt_dict['f_vals'].extend(lag_vals)
        plt_dict['augmented_grads'].extend([np.linalg.norm(g) for g in grad_vals])
        plt_dict['max_violations'].extend([np.max([constr(x) for constr in lagrangian.constraints] + [0])] * len(x_vals))

    return x, lambda_, plt_dict

if __name__ == "__main__":

    f = f()
    constraint_funcs = [g1(), g2(), g3()]
    phi_p = phi_p()
    lagrangian = lagrangian(f, constraint_funcs, phi_p, 1)

    # parameters
    x0 = np.array([0, 0])
    lambda_0 = np.array([1.0, 1.0, 1.0])

    # optimal analytically achived solution
    x_opt = np.array([2/3, 2/3])
    lambda_opt = np.array([12, 11+1/3, 0])
    f_opt = lagrangian(x_opt, lambda_opt)

    # numeric optimization
    optimal_x, lambda_, plt_dict = augmented_lagrangian(lagrangian, x0, lambda_0, p_max=100, alpha=2, max_iter=20)

    print("optimal x: ", optimal_x)
    print("optimal lambda: ", lambda_)

    fig, ax = plt.subplots(2,2)
    iters = list(range(plt_dict['newton_iter']))
    ax[0, 0].plot(iters, plt_dict['augmented_grads'])
    ax[0, 0].set(yscale="log", title="Augmented Lagrangian Gradient")

    ax[0, 1].plot(iters, np.abs(np.array(plt_dict['f_vals']) - f_opt))
    ax[0, 1].set(yscale="log", title="|f_k - f_opt|")

    ax[1, 0].plot(iters, plt_dict['max_violations'])
    ax[1, 0].set(yscale="log", title="Max Violation")

    x_dist = [np.linalg.norm(x - x_opt) for x in plt_dict['x_list']]
    lambda_dist = [np.linalg.norm(lam - lambda_opt) for lam in plt_dict['lam_list']]

    ax[1, 1].plot(iters, lambda_dist, label=r'||$\lambda_k$ - $\lambda^*$||')
    ax[1, 1].plot(iters, x_dist, label=r'||$x_k$ - $x^*$||')
    ax[1, 1].set(yscale="log", title=r'Xs & $\lambda$s distance from optimum')
    ax[1, 1].legend()

    for ax in fig.axes:
        ax.set_xlabel("Newton Iteration")
    fig.tight_layout()
    plt.show()

