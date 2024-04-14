import numpy as np
from typing import Any, SupportsIndex


class MA:

    def __init__(
        self,
        y_vector,
        periods,
        acf_values,
        ar_coefficients,
        ar_weight=0.7,
        ma_weight=0.3,
    ):

        self.y_vector = np.array(y_vector, dtype=np.float64)
        self.start_index = np.max(periods)
        self.index_vector = np.arange(self.start_index, len(self.y_vector))
        self.periods = np.array(periods)

        self.ar_coefficients = ar_coefficients
        self.ma_coefficients = self._get_ma_coefficients(acf_values)

        self.ar_predicted_value_vector = np.zeros(len(self.y_vector))
        self.arma_prediction_value_vector = np.zeros(len(self.y_vector))

        self.ar_weight = ar_weight
        self.ma_weight = ma_weight

    def _get_predicted_value(self, index):
        true_y_values = self.y_vector[index - self.periods]
        arma_y_values = self.arma_prediction_value_vector[index - self.periods]

        ar = np.sum(self.ar_coefficients * true_y_values)
        ma = np.sum((true_y_values - arma_y_values) * self.ma_coefficients)
        y = ma * self.ma_weight + ar * self.ar_weight
        self.arma_prediction_value_vector[index] = y

    def _get_ma_coefficients(self, acf_values):
        numerator = acf_values
        denominator = np.sqrt(np.sum(acf_values**2))
        return numerator / denominator

    def _run_vectorized(self, iterator, function):
        vectorized_function = np.vectorize(function)
        return vectorized_function(iterator)

    def set_up_initial_arma_predictions(self):
        self.arma_prediction_value_vector[: self.start_index + 1] = self.y_vector[
            : self.start_index + 1
        ]

    def update_arma_predictions(self):
        self.set_up_initial_arma_predictions()
        self._run_vectorized(self.index_vector, self._get_predicted_value)

    def get_arma_predictions(self):
        return self.arma_prediction_value_vector

    def _get_rmse(self):
        residuals = (
            self.y_vector[self.start_index :]
            - self.arma_prediction_value_vector[self.start_index :]
        ) ** 2
        residual_sum = np.sum(residuals)
        rmse = np.sqrt(residual_sum) / len(residuals)

        return rmse


# class TailedQueue(np.ndarray):

#     def __new__(self, arr, max_size):
#         self.max_size = max_size
#         self.tail = max_size - 1
#         obj = np.asarray(arr).view(self)
#         return obj

#     def enqueue(self, value):
#         self.tail = (self.tail + 1) % self.max_size
#         self[self.tail] = value
#         return self

#     def __getitem__(self, index):
#         index = (self.tail + 1 + index) % self.max_size
#         return super(TailedQueue, self).__getitem__(index)
