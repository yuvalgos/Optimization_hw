import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append('..')

#### parameters
eps = 1e-5
c_1 = 0.25
c_2 = 0.9


class f_xexp:
    # Q1.3.5
    def __call__(self, x):
        x = x.reshape(-1,2)
        x1, x2 = x[:,0], x[:,1]
        return x1 * np.exp(-1*(x1**2 + x2**2))


class activation_function:
    # Q 1.3.6
    def __call__(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def grad(self, x):
        return 1 - self.__call__(x) ** 2


class loss_function:
    # Q 1.3.7
    def __call__(self, y, y_hat):
        return (y - y_hat) ** 2

    def grad(self, y, y_hat):
        return 2 * (y - y_hat)


def generate_dataset(n):
    """
    samples dataset set from function
    :param n: dataset size
    :return:
    """
    # default rand samples on [0,1) so extending to [-2,2)
    x_ds = np.random.rand(n, 2) * 4 - 2
    fn = f_xexp()
    y_ds = fn(x_ds)
    return x_ds, y_ds


def init_nn_params():
    w1 = np.random.randn(2, 4).reshape(-1) / 2
    w2 = np.random.randn(4, 3).reshape(-1) / np.sqrt(3)
    w3 = np.random.randn(3, 1).reshape(-1)
    b1 = np.zeros(4)
    b2 = np.zeros(3)
    b3 = np.zeros(1)

    W = np.concatenate((w1, b1, w2, b2, w3, b3))
    return W


def nn_forward(x, W):
    # make sure x is column vector:
    if x.shape == (2,):
        x = x.reshape(-1, 1)

    w1 = W[:8].reshape(2, 4)
    b1 = W[8:12].reshape(4, 1)
    w2 = W[12:24].reshape(4, 3)
    b2 = W[24:27].reshape(3, 1)
    w3 = W[27:30].reshape(3, 1)
    b3 = W[30]

    act_fn = activation_function()
    z1 = w1.T @ x + b1
    a1 = act_fn(z1)
    z2 = w2.T @ a1 + b2
    a2 = act_fn(z2)
    z3 = w3.T @ a2 + b3
    y_hat = z3

    return y_hat


def nn_loss_entire_set(x_arr, y_arr, W):
    y_hat_arr = []
    for x in x_arr:
        y_hat_arr.append(nn_forward(x, W))

    y_hat_arr = np.array(y_hat_arr)
    return np.mean(loss_function()(y_arr, y_hat_arr))


def nn_grad(x, y, W):
    """w.r to the loss"""
    # make sure x is column vector:
    if x.shape == (2,):
        x = x.reshape(-1, 1)

    w1 = W[:8].reshape(2, 4)
    b1 = W[8:12].reshape(4, 1)
    w2 = W[12:24].reshape(4, 3)
    b2 = W[24:27].reshape(3, 1)
    w3 = W[27:30].reshape(3, 1)
    b3 = W[30]

    act_fn = activation_function()
    loss_fn = loss_function()

    # forward pass
    z1 = w1.T @ x + b1
    a1 = act_fn(z1)
    z2 = w2.T @ a1 + b2
    a2 = act_fn(z2)
    z3 = w3.T @ a2 + b3
    y_hat = z3

    loss = loss_fn(y, y_hat)

    # backward pass
    loss_grad = loss_fn.grad(y, y_hat)
    w3_grad = a2 * loss_grad
    b3_grad = loss_grad

    a2_grad = loss_grad * w3
    w2_grad = a1 @ a2_grad.T @ np.diag(act_fn.grad(z2).squeeze())
    b2_grad = a2_grad.T @ np.diag(act_fn.grad(z2).squeeze())

    a1_grad = a2_grad.T @ (np.diag(act_fn.grad(z2).squeeze()) @ w2.T)
    w1_grad = x @ a1_grad @ np.diag(act_fn.grad(z1).squeeze())
    b1_grad = a1_grad @ np.diag(act_fn.grad(z1).squeeze())

    W = np.concatenate([w1_grad.reshape(-1),
                        b1_grad.squeeze(),
                        w2_grad.reshape(-1),
                        b2_grad.squeeze(),
                        w3_grad.reshape(-1),
                        b3_grad.reshape(-1)])

    return W


def nn_grad_entire_set(x_arr, y_arr, W):
    grad = np.zeros_like(W)
    for x, y in zip(x_arr, y_arr):
        grad += nn_grad(x, y, W)
    grad = grad / x_arr.shape[0]
    return grad


def inexact_line_search(x_train, y_train, W, p, grad, alpha, beta):
    """
    Inexact Line Search
    from wikipedia:
    sigma is c
    m is grad^T @ p
    """
    print("starting line search")

    loss_orig = nn_loss_entire_set(x_train, y_train, W)
    loss_new = nn_loss_entire_set(x_train, y_train, W + alpha * p)
    grad_new = nn_grad_entire_set(x_train, y_train, W + alpha*p)
    while not (loss_new <= loss_orig + alpha * c_1 * p.T @ grad) \
            or not (grad_new.T @ p >= c_2 * grad.T @ p):
        alpha = beta * alpha

        loss_new = nn_loss_entire_set(x_train, y_train, W + alpha*p)
        grad_new = nn_grad_entire_set(x_train, y_train, W + alpha*p)

        print("alpha:", alpha)

    return alpha


def optimize_nn_params(x_train, y_train, W, eps, max_iter):
    """ with BFGS """
    grad = nn_grad_entire_set(x_train, y_train, W)
    H_inv = np.eye(grad.shape[0])
    p = - H_inv @ grad

    for i in range(max_iter):

        if np.linalg.norm(grad) < eps:
            break

        # line search
        alpha = inexact_line_search(x_train, y_train, W, p, grad, 1, 0.5)

        # update
        W_new = W + alpha * p
        grad_new = nn_grad_entire_set(x_train, y_train, W_new)
        y = grad_new - grad
        s = W_new - W
        rho = 1 / (y.T @ s)
        y, s = y.reshape(-1, 1), s.reshape(-1, 1)
        H_inv = (np.eye(W.shape[0]) - rho * s @ y.T) @ H_inv @ \
                (np.eye(W.shape[0]) - rho * y @ s.T) + rho * s @ s.T
        p = - H_inv @ grad_new

        # update
        W = W_new
        grad = grad_new

        # print
        print('iter: {}, grad_norm: {}, loss: {}'.format(i,
                                                         np.linalg.norm(grad),
                                                         nn_loss_entire_set(x_train, y_train, W)))
    return W


if __name__ == '__main__':
    # Q1.3.10
    x_train, y_train = generate_dataset(n=500)
    x_test, y_test = generate_dataset(n=200)

    for eps in [1e-1, 1e-2, 1e-3, 1e-4]:
        W0 = init_nn_params()
        W_opt = optimize_nn_params(x_train, y_train, W0, eps, max_iter=10000)

        loss = nn_loss_entire_set(x_test, y_test, W_opt)
        print(loss)


    # nn = NeuralNetwork(d_in=2)
