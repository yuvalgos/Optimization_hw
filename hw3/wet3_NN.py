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
    w1 = np.random.rand(2, 4).reshape(-1)
    w2 = np.random.rand(4, 3).reshape(-1)
    w3 = np.random.rand(3, 1).reshape(-1)
    b1 = np.zeros(4)
    b2 = np.zeros(3)
    b3 = np.zeros(1)

    W = np.concatenate((w1, b1, w2, b2, w3, b3))
    return W


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
    w3_grad = a2 * loss_grad * z3
    b3_grad = loss_grad * z3

    w2_grad = a1 @ w3_grad.T @ np.diag(act_fn.grad(z2).squeeze())
    b2_grad = w3_grad.T @ np.diag(act_fn.grad(z2).squeeze())

    w1_grad = x @ w2_grad.T @ np.diag(act_fn.grad(z1).squeeze())
    b1_grad = w2_grad.T @ np.diag(act_fn.grad(z1).squeeze())

    l = 0



# def optimize_nn(x_train, y_train, nn, epsilon):


if __name__ == '__main__':
    x = np.array([1, 2])
    y = 5
    W = np.zeros([31])
    nn_grad(x, y, W)

    # Q1.3.10
    x_train, y_train = generate_dataset(n=500)
    x_test, y_test = generate_dataset(n=200)

    # nn = NeuralNetwork(d_in=2)
