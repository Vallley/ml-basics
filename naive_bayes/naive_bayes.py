import math
import numpy as np


class NaiveBayes:
    #def __init__(self):
        #self.categories_list = list(categories)

    def fit(self, x, y):
        self.categories = set(y)
        self.num_docs = x.shape[0]
        self.num_features = x.shape[1]
        self.classes_stat = {}
        self.indexes = {}
        self.words_per_class = {}
        self.word_freqs_per_class = {}
        self.words_per_class2 = {}
        for category in self.categories:
            self.classes_stat[category] = np.sum(np.array(y) == category)  # число объектов каждого класса
            self.indexes[category] = np.where(np.array(y) == category)[0]  # индексы текстов по категориям
            self.word_freqs_per_class[category] = np.sum(x[self.indexes[category]].toarray(),
                                                         axis=0)  # частота использования слов в каждом классе
            invalid_indices = np.where(self.word_freqs_per_class[category] == 0)
            self.words_per_class[category] = len(
                np.delete(self.word_freqs_per_class[category], invalid_indices))  # количество слов в каждом классе

    def predict(self, x):
        pred_per_class = {}
        x = x.toarray()
        i = 0
        pred = []
        for doc in x:
            pred_per_class[i] = {}
            for category in self.categories:
                word_stat = 0
                words_indexes = np.nonzero(doc)
                for word in words_indexes[0]:
                    word_stat += math.log((self.word_freqs_per_class[category][word] + 1) / (
                                self.num_features + self.words_per_class[category]))
                # print(word_stat)
                pred_per_class[i][category] = math.log(self.classes_stat[category] / self.num_docs) + word_stat
            i += 1
        for key, value in pred_per_class.items():
            # category = str(*[category for category, stat in value.items() if stat == min(value.values())])
            # pred.append(self.categories_list.index(category))
            pred.append(*[category for category, stat in value.items() if stat == max(value.values())])
        return pred
