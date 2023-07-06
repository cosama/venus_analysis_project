import math

import numpy as np
from sklearn.neighbors import KDTree



class KNNRegressor:
    def __init__(self, to_normalize, num_neighbors, weights=1):
        self.num_neighbors = num_neighbors
        self.to_normalize = to_normalize
        self.maximum = None
        self.minimum = None
        self.weights = weights

    def fit(self, x_data, y_data):
        if None == self.maximum:
            self.maximum = np.max(x_data)
        else:
            self.maximum = np.max(self.maximum, np.max(x_data))

        if None == self.minimum:
            self.minimum = np.min(x_data)
        else:
            self.minimum = np.min(self.minimum, np.min(x_data))

        self.kd_tree = KDTree(self.weigh(self.normalize(x_data)))
        self.y_data = y_data


    def normalize(self, data):
        if self.to_normalize:
            data = (data - self.minimum) / (self.maximum - self.minimum)
        return (data)


    def weigh(self, data):
        # we want certain dimensions to be more imporant than others
        # so we weigh them
        return self.weights * data


    def predict(self, data):
        _, indices = self.kd_tree.query(self.weigh(self.normalize(data)).reshape(1, -1), k=self.num_neighbors)
        k_nearest_labels = self.y_data[indices.flatten()]
        return np.mean(k_nearest_labels)
