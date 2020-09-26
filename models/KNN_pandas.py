from collections import Counter
import numpy as np
import pandas as pd

class KNN:
    def __init__(self, num_neighbors, data, attribute_names):
        self.num_neighbors = num_neighbors
        self.data = data
        self.attribute_names = attribute_names

    def find_nearest_neighbors(self, query_point):
        results = self.calculate_distances(query_point)
        results = results.sort_values(by='distances')
        print(results.head(self.num_neighbors))
        nearest_neighbors = list(results.head(self.num_neighbors)['Class'])
        return nearest_neighbors

    def calculate_distances(self, query_point):
        vectorized_distance = np.vectorize(self.distance_helper)
        results = pd.DataFrame(vectorized_distance(query_point.iloc[1:-1], self.data.iloc[:, 1:-1]))
        results_column_names = []

        for name in self.attribute_names[1:-1]:
            results_column_names.append(name + '_dist')

        results.columns = results_column_names
        results['Class'] = self.data['Class'].values
        results['sum_squared_diffs'] = results.iloc[:, :-1].sum(axis=1)
        results['distances'] = results['sum_squared_diffs'].apply(lambda x: np.sqrt(x))
        return results

    def distance_helper(self, x, y):
        return np.power(x - y, 2)

    def classify(self, query_point):
        nearest_neighbors = self.find_nearest_neighbors(query_point)
        class_label = Counter(nearest_neighbors).most_common(1)
        return class_label[0][0]
