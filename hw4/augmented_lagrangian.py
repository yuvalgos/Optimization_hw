import numpy as np
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
    plt_dict = dict(augmented_grads=[], f_vals=[], max_violations=[], x_list=[], lam_list=[])

    for i in range(max_iter):
        new_x, x_vals, lag_vals = newton_method(lagrangian, x, lambda_)

        new_lambda = np.zeros(len(lambda_))
        for j in range(len(lambda_)):
            new_lambda[j] = lambda_[j] * lagrangian.phi_p.derivative(lagrangian.constraints[j](new_x), lagrangian.p)

        lagrangian.p = min(p_max, alpha * lagrangian.p)

        x = new_x
        lambda_ = new_lambda

        plt_dict['x_list'].append(x)
        plt_dict['lam_list'].append(lambda_)
        plt_dict['f_vals'].append(lagrangian(x, lambda_))
        plt_dict['augmented_grads'].append(np.linalg.norm(lagrangian.grad(x, lambda_)))
        plt_dict['max_violations'].append(np.max(lagrangian.constraints(x)))

        print(lambda_)

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
    x_opt = np.array([[2/3], [2/3]])
    lambda_opt = np.array([[12], [11+1/3], [0]])
    f_opt = langrangian(x_opt, lambda_opt)

    # numeric optimization
    optimal_x, lambda_, plt_dict = augmented_lagrangian(lagrangian, x0, lambda_0, p_max=1000, alpha=2, max_iter=10)

    fig, ax = plt.subplots(2,2)
    iters = np.arange(len(plt_dict['x_list']))
    ax[0,0].plot(iters, plt_dict['augmented_grads'])
    ax[0.0].set(yscale="log", title="Augmented Lagrangian Gradient")

    ax[0,1].plot(iters, np.abs(np.array(plt_dict['f_vals']) - f_opt))
    ax[0,1].set(yscale="log", title="|f_k - f_opt|")

    ax[1,0].plot(iters, plt_dict['max_violations'])
    ax[1,0].set(yscale="log", title="Max Violation")

    ax[1,1].plot(iters, plt_dict['lam_list'], label=r'||$\lambda_k$ - $\lambda^*$||')
    ax[1,1].plot(iters, plt_dict['x_list'], label=r'||$\x_k$ - $\x^*$||')
    ax[1,1].set(yscale="log", title=r'Xs & $\lambda$s distance from optimum')
    plt.show()

    print("optimal x: ", optimal_x)
    print("optimal lambda: ", lambda_)
