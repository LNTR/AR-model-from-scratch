# used for cleaning and transforming the inputs inside the excel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


def weighted_moving_average(array, window=5):
    weights = [(1 / window) for _ in range(1, window + 1)]
    offset_count = math.floor(window / 2)
    array = np.insert(array, 0, [array[0] for _ in range(offset_count)])
    array = np.insert(array, 0, [array[-1] for _ in range(offset_count)])
    if not (window % 2):
        array = np.delete(array, [-1])
    return np.convolve(array, weights, mode="valid")


class LinearRegression:
    def __init__(self, x_vector, y_vector):

        self.x_vector = np.array(x_vector, dtype=np.float64)
        self.y_vector = np.array(y_vector, dtype=np.float64)
        self.w0 = 0
        self.w1 = 0

    def _get_predicted_value_matrix(self, x):
        formula = lambda x: self.w0 + self.w1 * x
        return formula(x)

    def _get_gradient_matrix(self):
        predictions = self._get_predicted_value_matrix(self.x_vector)
        w0_hat = sum((self.y_vector - predictions))
        w1_hat = sum((self.y_vector - predictions) * self.x_vector)

        gradient_matrix = np.array([w0_hat, w1_hat])
        gradient_matrix = -2 * gradient_matrix

        return gradient_matrix

    def fit(self, step_size=0.00001, num_iterations=500):
        for i in range(1, num_iterations):
            gradient_matrix = self._get_gradient_matrix()
            self.w0 -= step_size * (gradient_matrix[0])
            self.w1 -= step_size * (gradient_matrix[1])
            self.show_coeffiecients()

    def show_coeffiecients(self):
        print(f"w0: {self.w0}\tw1: {self.w1}\t")

    def predict(self, x):
        y = self.w0 + self.w1 * x
        return y


x = [x for x in range(-12, 12)]
f = lambda x: 5 * x - 7
y = [f(x_val) for x_val in x]

model = LinearRegression(x, y)
model.fit(num_iterations=200000)

model.show_coeffiecients()

# plt.plot(x, y)
# plt.plot(x, model.predict(np.array(x)))
# model.show_coeffiecients()
# plt.show()
