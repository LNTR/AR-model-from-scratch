import numpy as np
import pandas as pd


# Assumption : Since there's a possible seasonality and complex trend, assumed that the polynomial function is in the form
# y=w0 + w1.x + w2.x.x + w3.x.x.x + w4.sin(x) + w5.cos(x)


def weighted_moving_average(a, window=3):
    weights = [(1 / window) for _ in range(1, window + 1)]
    return np.convolve(a, weights, mode="valid")


class PolynomialRegression:
    def __init__(self, x_vector, y_vector):
        self.x_vector = x_vector
        self.y_vector = y_vector

        self.w0 = 0
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        self.w4 = 0
        self.w5 = 0

        self.l1 = 0
        self.tollerance = 0.00001

    def set_l1(self, l1):
        self.l1 = l1

    def _get_predicted_value_matrix(self, x):
        formula = (
            lambda x: self.w0
            + self.w1 * x
            + self.w2 * x * x
            + self.w3 * x * x * x
            + self.w4 * np.sin(x)
            + self.w5 * np.cos(x)
        )
        return formula(x)

    def _get_gradient_matrix(self):
        gradient_functions = np.vectorize(
            lambda x, y: (
                sum((y - self._get_predicted_value_matrix(x))),
                sum((y - self._get_predicted_value_matrix(x)) * x),
                sum((y - self._get_predicted_value_matrix(x)) * (x**2)),
                sum((y - self._get_predicted_value_matrix(x)) * (x**3)),
                sum((y - self._get_predicted_value_matrix(x)) * np.cos(x)),
                sum((y - self._get_predicted_value_matrix(x) * np.sin(x) * (-1))),
            )
        )

        gradient_matrix = (
            np.array(gradient_functions(self.x_vector, self.y_vector)) * -2
        )

        return gradient_matrix

    def fit(self, learning_rate=0.01, num_iterations=1000):
        pass

    def predict(self, x):
        y = (
            self.w0
            + self.w1 * x
            + self.w2 * x * x
            + self.w3 * x * x * x
            + self.w4 * np.sin(x)
            + self.w5 * np.cos(x)
        )
        return y
