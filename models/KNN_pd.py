from collections import Counter
import numpy as np
import pandas as pd


class KNN:
    def __init__(self, num_neighbors, data, attribute_names, bandwidth=None, error=None, binary_classification=False):
        self.num_neighbors = num_neighbors
        self.data = data
        self.attribute_names = attribute_names
        self.bandwidth = bandwidth
        self.error = error
        self.training_stats = {
            'correct': 0,
            'incorrect': 0,
            'total_predictions': 0,
            'success_rate': 0,
            'mean_squared_error': 0,
            # Used in binary classification:
            'false_positive': 0,
            'false_negative': 0,
        }
        self.parameter_tuning = {}
        self.test_stats = {
            'correct': 0,
            'incorrect': 0,
            'total_predictions': 0,
            'success_rate': 0,
            'mean_squared_error': 0,
            # Used in binary classification:
            'false_positive': 0,
            'false_negative': 0,
        }
        self.binary_classification = binary_classification

    def train(self, regression=False):
        if not regression:
            self.data.apply(self.classify, axis=1)
        else:
            self.data.apply(self.regress, axis=1)
            self.training_stats['mean_squared_error'] /= self.training_stats['total_predictions']

    def test(self, test_set, regression=False):
        if not regression:
            test_set.apply(self.classify, testing=True, axis=1)
        else:
            self.data.apply(self.regress, testing=True, axis=1)
            self.test_stats['mean_squared_error'] /= self.test_stats['total_predictions']

    def tune(self, tuning_set, parameters, regression=False):
        for parameter in parameters:
            self.num_neighbors = parameter
            self.parameter_tuning[parameter] = {
                'correct': 0,
                'incorrect': 0,
                'total_predictions': 0,
                'success_rate': 0,
                'mean_squared_error': 0,
                # Used in binary classification:
                'false_positive': 0,
                'false_negative': 0,
            }
            if not regression:
                tuning_set.apply(self.classify, parameter=parameter, tuning=True, axis=1)
            else:
                tuning_set.apply(self.regress, parameter=parameter, tuning=True, axis=1)
                self.parameter_tuning[parameter]['mean_squared_error'] /= \
                    self.parameter_tuning[parameter]['total_predictions']

        optimal_parameter, max_success_rate = 1, 0
        for k, v in self.parameter_tuning.items():
            if v['success_rate'] > max_success_rate:
                optimal_parameter = k
                max_success_rate = v['success_rate']
        self.num_neighbors = optimal_parameter
        return optimal_parameter

    def find_nearest_neighbors(self, query_point):
        results = self.calculate_distances(query_point)
        results = results.sort_values(by='distances')
        # print(results.head(self.num_neighbors))
        nearest_neighbors = results.head(self.num_neighbors)
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

    def classify(self, query_point, parameter=None, tuning=False, testing=False):
        nearest_neighbors = list(self.find_nearest_neighbors(query_point)['Class'])
        nearest_neighbors = Counter(nearest_neighbors).most_common(1)
        prediction = nearest_neighbors[0][0]
        if not tuning and not testing:
            self.update_training_stats(prediction, query_point.iloc[-1])
        elif tuning:
            self.update_tuning_stats(prediction, parameter, query_point.iloc[-1])
        elif testing:
            self.update_testing_stats(prediction, query_point.iloc[-1])
        return prediction

    def regress(self, query_point, parameter=None, tuning=False, testing=False):
        nearest_neighbors = self.find_nearest_neighbors(query_point)
        nearest_distances = nearest_neighbors['distances']
        nearest_values = nearest_neighbors['Class']
        weighted_sum, normalizer = 0, 0
        for distance, value in zip(nearest_distances, nearest_values):
            weighted_sum += np.exp(-(distance**2/(2*self.bandwidth))) * value
            normalizer += np.exp(-(distance**2/(2*self.bandwidth)))
        prediction = weighted_sum / normalizer
        squared_error = np.power(query_point.iloc[-1] - prediction, 2)
        if not tuning and not testing:
            self.update_training_stats(prediction, query_point.iloc[-1], error=squared_error, regression=True)
        elif tuning:
            self.update_tuning_stats(prediction, parameter, query_point.iloc[-1], error=squared_error, regression=True)
        elif testing:
            self.update_testing_stats(prediction, query_point.iloc[-1], error=squared_error, regression=True)
        # correct = "Correct" if squared_error < self.error else "Incorrect"
        # print(f'{correct} -- Prediction: {prediction}, Actual: {query_point.iloc[-1]}')
        # print(f'{correct} -- Error: {squared_error}')
        return prediction

    def update_training_stats(self, prediction, real_class, error=None, regression=False):
        if not regression:
            correct = prediction == real_class
        else:
            correct = error < self.error
        if correct:
            self.training_stats['correct'] += 1
        else:
            self.training_stats['incorrect'] += 1
            if self.binary_classification:
                if prediction and not real_class:
                    self.training_stats['false_positive'] += 1
                elif not prediction and real_class:
                    self.training_stats['false_negative'] += 1
        self.training_stats['total_predictions'] += 1
        self.training_stats['success_rate'] = self.training_stats['correct'] / self.training_stats['total_predictions']
        if regression:
            self.training_stats['mean_squared_error'] += error

    def update_tuning_stats(self, prediction, parameter, real_class, error=None, regression=False):
        if not regression:
            correct = prediction == real_class
        else:
            correct = error < self.error
        if correct:
            self.parameter_tuning[parameter]['correct'] += 1
        else:
            self.parameter_tuning[parameter]['incorrect'] += 1
            if self.binary_classification:
                if prediction and not real_class:
                    self.parameter_tuning[parameter]['false_positive'] += 1
                elif not prediction and real_class:
                    self.parameter_tuning[parameter]['false_negative'] += 1
        self.parameter_tuning[parameter]['total_predictions'] += 1
        self.parameter_tuning[parameter]['success_rate'] = \
            self.parameter_tuning[parameter]['correct'] / self.parameter_tuning[parameter]['total_predictions']
        if regression:
            self.parameter_tuning[parameter]['mean_squared_error'] += error

    def update_testing_stats(self, prediction, real_class, error=None, regression=False):
        if not regression:
            correct = prediction == real_class
        else:
            correct = error < self.error
        if correct:
            self.test_stats['correct'] += 1
        else:
            self.test_stats['incorrect'] += 1
            if self.binary_classification:
                if prediction and not real_class:
                    self.test_stats['false_positive'] += 1
                elif not prediction and real_class:
                    self.test_stats['false_negative'] += 1
        self.test_stats['total_predictions'] += 1
        self.test_stats['success_rate'] = self.test_stats['correct'] / self.test_stats['total_predictions']
        if regression:
            self.test_stats['mean_squared_error'] += error

    def report_training_stats(self):
        print("Training stats: \n", self.training_stats)

    def report_tuning_stats(self):
        print("\nTuning Stats: ")
        for parameter, stats in self.parameter_tuning.items():
            print(parameter, stats)

    def report_testing_stats(self):
        print("\nTesting stats: \n", self.test_stats)
