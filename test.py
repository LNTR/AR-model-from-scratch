import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


def weighted_moving_average(a, window=3):
    weights = [(1 / window) for _ in range(1, window + 1)]
    return np.convolve(a, weights, mode="valid")


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


def weighted_moving_average(a, window=3):
    weights = [(1 / window) for _ in range(1, window + 1)]
    return np.convolve(a, weights, mode="valid")


class LinearRegression:
    def __init__(self, x_vector, y_vector):
        self.x_vector = np.array(x_vector)
        self.y_vector = np.array(y_vector)

        self.w0 = 0
        self.w1 = 0

    def _get_predicted_value_matrix(self, x):
        formula = lambda x: self.w0 + self.w1 * x
        return formula(x)

    def _get_gradient_matrix(self):
        gradient_matrix = [
            sum((self.y_vector - self._get_predicted_value_matrix(self.x_vector))),
            sum(
                (self.y_vector - self._get_predicted_value_matrix(self.x_vector))
                * self.x_vector
            ),
        ]

        gradient_matrix = np.array(gradient_matrix) * -2

        return gradient_matrix

    def fit(self, step_size=0.001, num_iterations=10000):
        for i in range(1, num_iterations):
            gradient_matrix = self._get_gradient_matrix()
            self.w0 -= step_size * gradient_matrix[0]
            self.w1 -= step_size * gradient_matrix[1]

    def _show_coeffiecients(self):
        print(f"w0: {self.w0}\nw1: {self.w1}\n")

    def predict(self, x):
        y = self.w0 + self.w1 * x
        return y


x = [
    1,
    2,
    3,
]
y = [6, 9, 12]
model = LinearRegression(x, y)
model.fit(num_iterations=100000)

model._show_coeffiecients()
# plt.plot(x, y)
# plt.plot(x, model.predict(np.array(x)))

# plt.show()
