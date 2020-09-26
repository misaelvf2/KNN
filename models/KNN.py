from collections import Counter
import math

class KNN:
    def __init__(self, k, data, num_features):
        self.k = k
        self.data = data
        self.num_features = num_features

    def find_nearest_neighbors(self, query_point):
        neighbor_index_distance = []
        for index, example in enumerate(self.data):
            neighbor_index_distance.append((index, self.calculate_distance(query_point, example)))
        neighbor_index_distance.sort(key=lambda neighbor: neighbor[1])
        nearest_neighbors = []
        for index, distance in neighbor_index_distance[:self.k + 1]:
            nearest_neighbors.append(self.data[index])
        return nearest_neighbors

    def calculate_distance(self, query_point, data_point):
        distance = 0
        for feature in range(self.num_features):
            distance += math.pow(data_point[feature] - query_point[feature], 2)
        return math.sqrt(distance)


    def classify(self, query_point):
        nearest_neighbors = self.find_nearest_neighbors(query_point)
        labels = []
        for neighbor in nearest_neighbors:
            labels.append(neighbor[-1])
        mode = Counter(labels).most_common(1)
        return mode[0][0]
