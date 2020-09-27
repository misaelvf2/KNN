from collections import Counter
import numpy as np
import pandas as pd
from decimal import Decimal


class KNN:
    def __init__(self, num_neighbors, data, attribute_names, bandwidth=None, error=None):
        self.num_neighbors = num_neighbors
        self.data = data
        self.condensed_data = pd.DataFrame(columns=attribute_names)
        self.condensed_data = self.condensed_data.append(self.data.sample(frac=1, random_state=2).iloc[0, :])
        self.attribute_names = attribute_names
        self.bandwidth = bandwidth
        self.error = error
        self.training_stats = {
            'correct': 0,
            'incorrect': 0,
            'total_predictions': 0,
            'success_rate': 0,
            'mean_squared_error': 0,
        }
        self.k_tuning = {}
        self.bandwidth_tuning = {}
        self.test_stats = {
            'correct': 0,
            'incorrect': 0,
            'total_predictions': 0,
            'success_rate': 0,
            'mean_squared_error': 0,
        }

    def train(self, edited=False, regression=False):
        if not regression:
            self.data.apply(self.classify, edited=edited, axis=1)
        else:
            self.data.apply(self.regress, edited=edited, axis=1)
            self.training_stats['mean_squared_error'] /= self.training_stats['total_predictions']

    def test(self, test_set, regression=False):
        if not regression:
            test_set.apply(self.classify, testing=True, axis=1)
        else:
            self.data.apply(self.regress, testing=True, axis=1)
            self.test_stats['mean_squared_error'] /= self.test_stats['total_predictions']

    def tune_k(self, tuning_set, parameters, regression=False):
        for parameter in parameters:
            self.num_neighbors = parameter
            self.k_tuning[parameter] = {
                'correct': 0,
                'incorrect': 0,
                'total_predictions': 0,
                'success_rate': 0,
                'mean_squared_error': 0,
            }
            if not regression:
                tuning_set.apply(self.classify, parameter=parameter, tuning='k', axis=1)
            else:
                tuning_set.apply(self.regress, parameter=parameter, tuning='k', axis=1)
                self.k_tuning[parameter]['mean_squared_error'] /= \
                    self.k_tuning[parameter]['total_predictions']

        optimal_parameter, max_success_rate = parameters[0], 0
        for k, v in self.k_tuning.items():
            if v['success_rate'] > max_success_rate:
                optimal_parameter = k
                max_success_rate = v['success_rate']
        self.num_neighbors = optimal_parameter
        return optimal_parameter

    def tune_bandwidth(self, tuning_set, parameters):
        for parameter in parameters:
            self.bandwidth = parameter
            self.bandwidth_tuning[parameter] = {
                'correct': 0,
                'incorrect': 0,
                'total_predictions': 0,
                'success_rate': 0,
                'mean_squared_error': 0,
            }
            tuning_set.apply(self.regress, parameter=parameter, tuning='bandwidth', axis=1)
            self.bandwidth_tuning[parameter]['mean_squared_error'] /= \
                self.bandwidth_tuning[parameter]['total_predictions']

        optimal_parameter, max_success_rate = parameters[0], 0
        for k, v in self.bandwidth_tuning.items():
            if v['success_rate'] > max_success_rate:
                optimal_parameter = k
                max_success_rate = v['success_rate']
        self.bandwidth = optimal_parameter
        return optimal_parameter

    def find_nearest_neighbors(self, query_point, condensed=False):
        results = self.calculate_distances(query_point, condensed=condensed)
        results = results.sort_values(by='distances')
        if not condensed:
            nearest_neighbors = results.head(self.num_neighbors)
        else:
            nearest_neighbors = results.head(1)
        return nearest_neighbors

    def calculate_distances(self, query_point, condensed=False):
        vectorized_distance = np.vectorize(self.distance_helper)
        if not condensed:
            results = pd.DataFrame(vectorized_distance(query_point.iloc[1:-1], self.data.iloc[:, 1:-1]))
        else:
            results = pd.DataFrame(vectorized_distance(query_point.iloc[1:-1], self.condensed_data.iloc[:, 1:-1]))
        results_column_names = []

        for name in self.attribute_names[1:-1]:
            results_column_names.append(name + '_dist')

        results.columns = results_column_names
        if not condensed:
            results['Class'] = self.data['Class'].values
        else:
            results['Class'] = self.condensed_data['Class'].values
        results['sum_squared_diffs'] = results.iloc[:, :-1].sum(axis=1)
        results['distances'] = results['sum_squared_diffs'].apply(lambda x: np.sqrt(x))
        return results

    def distance_helper(self, x, y, norm='Euclidean'):
        if norm == 'Euclidean':
            return np.power(x - y, 2)

    def classify(self, query_point, parameter=None, edited=False, condensed=False, tuning=False, testing=False):
        nearest_neighbors = list(self.find_nearest_neighbors(query_point, condensed=condensed)['Class'])
        nearest_neighbors = Counter(nearest_neighbors).most_common(1)
        prediction = nearest_neighbors[0][0]
        if not tuning and not testing:
            self.update_training_stats(prediction, query_point.iloc[-1], condensed=condensed,
                                       query_point=query_point, edited=edited)
        elif tuning == 'k':
            self.update_k_tuning_stats(prediction, parameter, query_point.iloc[-1])
        elif tuning == 'bandwidth':
            self.update_bandwidth_tuning_stats(prediction, parameter, query_point.iloc[-1])
        elif testing:
            self.update_testing_stats(prediction, query_point.iloc[-1])
        return prediction

    def regress(self, query_point, parameter=None, tuning=False, edited=False, condensed=False, testing=False):
        nearest_neighbors = self.find_nearest_neighbors(query_point, condensed=condensed)
        nearest_distances = nearest_neighbors['distances']
        nearest_values = nearest_neighbors['Class']
        weighted_sum, normalizer = 0, 0
        for distance, value in zip(nearest_distances, nearest_values):
            arg = -(distance**2/(2*self.bandwidth))
            weighted_sum += np.exp(arg) * value
            normalizer += np.exp(arg)
        prediction = weighted_sum / normalizer
        # print(f'Prediction: {prediction} -- Expected: {query_point.iloc[-1]}')
        squared_error = np.power(query_point.iloc[-1] - prediction, 2)
        if not tuning and not testing:
            self.update_training_stats(prediction, query_point.iloc[-1], error=squared_error,
                                       regression=True, edited=edited, condensed=condensed, query_point=query_point)
        elif tuning == 'k':
            self.update_k_tuning_stats(prediction, parameter, query_point.iloc[-1], error=squared_error, regression=True)
        elif tuning == 'bandwidth':
            self.update_bandwidth_tuning_stats(prediction, parameter, query_point.iloc[-1], error=squared_error,
                                       regression=True)
        elif testing:
            self.update_testing_stats(prediction, query_point.iloc[-1], error=squared_error, regression=True)
        return prediction

    def update_training_stats(self, prediction, real_class, query_point=None, edited=False, condensed=False,
                              error=None, regression=False):
        if not regression:
            correct = prediction == real_class
        else:
            correct = error < self.error
        if correct:
            self.training_stats['correct'] += 1
        else:
            if edited:
                self.data.drop(self.data.index[self.data['Id'] == query_point['Id']], inplace=True)
            elif condensed:
                self.condensed_data = self.condensed_data.append(query_point)
            self.training_stats['incorrect'] += 1
        self.training_stats['total_predictions'] += 1
        self.training_stats['success_rate'] = self.training_stats['correct'] / self.training_stats['total_predictions']
        if regression:
            self.training_stats['mean_squared_error'] += error

    def update_k_tuning_stats(self, prediction, parameter, real_class, error=None, regression=False):
        if not regression:
            correct = prediction == real_class
        else:
            correct = error < self.error
        if correct:
            self.k_tuning[parameter]['correct'] += 1
        else:
            self.k_tuning[parameter]['incorrect'] += 1
        self.k_tuning[parameter]['total_predictions'] += 1
        self.k_tuning[parameter]['success_rate'] = \
            self.k_tuning[parameter]['correct'] / self.k_tuning[parameter]['total_predictions']
        if regression:
            self.k_tuning[parameter]['mean_squared_error'] += error

    def update_bandwidth_tuning_stats(self, prediction, parameter, real_class, error=None, regression=False):
        if not regression:
            correct = prediction == real_class
        else:
            correct = error < self.error
        if correct:
            self.bandwidth_tuning[parameter]['correct'] += 1
        else:
            self.bandwidth_tuning[parameter]['incorrect'] += 1
        self.bandwidth_tuning[parameter]['total_predictions'] += 1
        self.bandwidth_tuning[parameter]['success_rate'] = \
            self.bandwidth_tuning[parameter]['correct'] / self.bandwidth_tuning[parameter]['total_predictions']
        if regression:
            self.bandwidth_tuning[parameter]['mean_squared_error'] += error

    def update_testing_stats(self, prediction, real_class, error=None, regression=False):
        if not regression:
            correct = prediction == real_class
        else:
            correct = error < self.error
        if correct:
            self.test_stats['correct'] += 1
        else:
            self.test_stats['incorrect'] += 1
        self.test_stats['total_predictions'] += 1
        self.test_stats['success_rate'] = self.test_stats['correct'] / self.test_stats['total_predictions']
        if regression:
            self.test_stats['mean_squared_error'] += error

    def report_training_stats(self):
        self.training_stats['set_size'] = len(self.data)
        print("Training stats: \n", self.training_stats)

    def report_k_tuning_stats(self):
        print("\nK tuning Stats: ")
        for parameter, stats in self.k_tuning.items():
            print(parameter, stats)

    def report_bandwidth_tuning_stats(self):
        print("\nBandwidth tuning Stats: ")
        for parameter, stats in self.bandwidth_tuning.items():
            print(parameter, stats)

    def report_testing_stats(self):
        print("\nTesting stats: \n", self.test_stats)

    def condense_data(self, regression=False):
        orig_size = len(self.data)
        prev_size = len(self.condensed_data)
        changed = True
        # self.data = self.data.sample(frac=1, random_state=2)
        while changed:
            if not regression:
                self.data.apply(self.classify, condensed=True, axis=1)
            else:
                self.data.apply(self.regress, condensed=True, axis=1)
                self.training_stats['mean_squared_error'] /= self.training_stats['total_predictions']
            changed = len(self.condensed_data) == prev_size
        self.data = self.condensed_data
        print(f'\nOriginal set size: {orig_size}, Condensed set size: {len(self.data)}')