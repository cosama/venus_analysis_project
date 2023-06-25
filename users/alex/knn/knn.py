import numpy as np
from sklearn.neighbors import KDTree



class KNNRegressionModel:
    def __init__(self, x_data, y_data, weights=1):
        self.y_data = y_data
        self.maximum = np.max(x_data)
        self.minimum = np.min(x_data)
        self.weights = weights
        self.kd_tree = KDTree(self.weigh(self.normalize(x_data)))


    def normalize(self, data):
        return (data - self.minimum) / (self.maximum - self.minimum)


    def weigh(self, data):
        # we want certain dimensions to be more imporant than others
        # so we weigh them
        return self.weights * data

    def predict(self, data, k=3):
        _, indices = self.kd_tree.query(self.weigh(self.normalize(data)).reshape(1, -1), k=k)
        k_nearest_labels = self.y_data[indices.flatten()]
        return np.mean(k_nearest_labels)
