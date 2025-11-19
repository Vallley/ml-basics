# %%
import sys
import numpy as np


def gini_index(y):
    label = set(y)
    y = np.array(y)
    counter = []
    for l in label:
        counter.append(np.where(y == l)[0].size / y.shape[0])
    gini = 1 - np.sum(np.array(counter) ** 2)
    return gini


def entropy_index(y):
    label = set(y)
    y = np.array(y)
    counter = []
    for l in label:
        counter.append(np.where(y == l)[0].size / y.shape[0])
    counter = np.array(counter)
    entropy = -np.sum(counter * np.log2(counter))
    return entropy


class TreeNode:
    def __init__(self, impurity=sys.float_info.max, target_value=None):
        self.left_child = None  # левое ответвление
        self.right_child = None  # правое ответвление
        self.is_leaf = True  # флаг, является ли этот узел терминльным (то есть листом)

        self.target_value = target_value  # значение целевого признака, которое предсказывает этот узел дерева

        self.condition_column = None  # id столбца, по которому будет делаться ветвление в этом узле дерева
        self.condition_value = None  # значение, величины, по которой было сделано ветвление
        self.impurity = impurity  # значение неопределённости для этого узла

    def predict(self, x):
        if self.is_leaf:
            return self.target_value

        mask = x[:, self.condition_column] >= self.condition_value

        res = np.zeros(x.shape[0])
        res[mask] = self.right_child.predict(x[mask])
        res[~mask] = self.left_child.predict(x[~mask])
        return res


def new_split_impurity(s_left, s_right, impurity_metric):
    if len(s_left) == 0 or len(s_right) == 0:
        return 1, 1, 1
    s = np.concatenate((s_left, s_right))
    impurity_right = impurity_metric(s_right)
    impurity_left = impurity_metric(s_left)
    impurity = s_left.shape[0] / s.shape[0] * impurity_left + s_right.shape[0] / s.shape[0] * impurity_right
    return impurity_left, impurity_right, impurity


def find_dominant_class(s):
    values, count = np.unique(s, return_counts=True)
    return values[np.argmax(count)]


def get_split_values(x):
    x = np.sort(x)
    split_values = []
    for i in range(len(x) - 1):
        split_values.append((x[i] + x[i + 1]) / 2)
    return list(set(split_values))


def find_best_split(x, y, impurity_metric):
    min_impurity = 1
    best_split_col = 0
    best_split_value = 0
    for col in range(x.shape[1]):
        split_values = get_split_values(x[:, col])
        for value in split_values:
            mask_left = x[:, col] <= value
            y_left = y[mask_left]
            y_right = y[~mask_left]
            impurity_left, impurity_right, impurity = new_split_impurity(y_left, y_right, impurity_metric)
            if impurity <= min_impurity:
                min_impurity = impurity
                best_split_col = col
                best_split_value = value
    return min_impurity, best_split_col, best_split_value


def build_next_node(x, y, impurity_metric):
    tree = TreeNode()
    if np.unique(y).shape[0] == 1 or len(x) == 1:
        tree.target_value = find_dominant_class(y)
        tree.is_leaf = True
        tree.impurity = 0
        return tree
    else:
        min_impurity, best_split_col, best_split_value = find_best_split(x, y, impurity_metric)
        if min_impurity < impurity_metric(y):
            mask = x[:, best_split_col] <= best_split_value
            tree.left_child = build_next_node(x[mask], y[mask], impurity_metric)
            tree.right_child = build_next_node(x[~mask], y[~mask], impurity_metric)
            tree.is_leaf = False
            tree.impurity = min_impurity
            tree.condition_column = best_split_col
            tree.condition_value = best_split_value
            tree.target_value = find_dominant_class(y)
        else:
            tree.target_value = find_dominant_class(y)
            tree.is_leaf = True
            tree.impurity = 0
            return tree
    return tree

# %%
