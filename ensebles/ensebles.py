# %%
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def MSE_loss(y_real, y_predicted):
    return 0.5 * (y_real - y_predicted) ** 2


def MSE_loss_gradient(y_real, y_predicted):
    return y_predicted - y_real


def MAE_loss(y_real, y_predicted):
    return abs(y_real - y_predicted)


def MAE_loss_gradient(y_real, y_predicted):
    return np.where(y_real >= y_predicted, 1, -1)


def lq_loss(y_real, y_predicted):
    return y_real * np.log(1 + np.exp(-y_predicted)) + (1 - y_real) * np.log(1 + np.exp(y_predicted))


def lq_loss_gradient(y_real, y_predicted):
    return 1 / (1 + np.exp(-y_predicted)) - y_real


losses = {
    'MSE': (MSE_loss, MSE_loss_gradient),
    'MAE': (MAE_loss, MAE_loss_gradient),
    'lq': (lq_loss, lq_loss_gradient),
}


class GradientTreeBoosting:
    def __init__(self, learning_rate=0.1, loss='MSE', iterations=10, **tree_kwargs):
        self.estimators = []
        self.tree_kwargs = tree_kwargs
        self.iterations = iterations
        self.loss_name = loss
        self.loss, self.loss_grad = losses.get(loss)
        self.learning_rate = learning_rate

    def set_initial_y(self, y, loss_name):
        if loss_name == 'MSE':
            self.initial_y = np.mean(y)
        elif loss_name == 'MAE':
            self.initial_y = np.median(y)
        elif loss_name == 'lq':
            p = np.mean(y)
            print(np.log(p / (1 - p)))
            self.initial_y = np.log(p / (1 - p))

    def fit(self, x, y):
        self.set_initial_y(y, self.loss_name)
        y_predicted = np.full_like(y, self.initial_y, dtype=float)
        for iteration in range(self.iterations):
            g = self.loss_grad(y, y_predicted)
            tree = DecisionTreeRegressor(**self.tree_kwargs)
            tree.fit(x, g)
            self.estimators.append(tree)
            y_predicted += tree.predict(x) * self.learning_rate

    def predict(self, x):
        result = np.full(x.shape[0], self.initial_y, dtype=float)
        for tree in self.estimators:
            result -= tree.predict(x) * self.learning_rate
        return result

# %%

#%%
