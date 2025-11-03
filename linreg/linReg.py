import numpy as np


class linReg:
    def __init__(self, lr=0.0001, iters=5000):
        self.lr = lr  # learning rate - шаг градиента функции
        self.iters = iters
        self.tolerance = 0.00001

    def create_array(self, x):
        x = np.array(x)
        if x.shape[1] > 1:
            x = x.transpose()
        else:
            x = x.reshape(-1, 1)
        x = np.hstack((x, np.ones((len(x), 1))))
        return x

    def find_weights(self, x, y):
        x = self.create_array(x)
        y = np.array(y)
        w = np.ones((len(x[0])))
        mse = 0
        for _ in range(self.iters):
            error = np.sum(x * w, axis=1) - y
            if abs(mse - np.sum(error ** 2) / len(x)) <= self.lr:
                self.w = w
                return w
            else:
                mse = np.sum(error ** 2) / len(x)
                for i in range(len(w)):
                    w[i] = w[i] - 2 * np.sum(error * x[:, i]) / len(x) * self.lr
        else:
            self.w = w
            return w

    def predict(self, x):
        x = self.create_array(x)
        return np.dot(x, self.w)
