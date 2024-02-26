import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


def weighted_moving_average(a, window=3):
    weights = [(1 / window) for _ in range(1, window + 1)]
    return np.convolve(a, weights, mode="valid")


class PolinomialRegression:
    def __init__(self, x_vector, y_vector):
        self.x_vector = np.array(x_vector)
        self.y_vector = np.array(y_vector)

        self.w0 = 0
        self.w1 = 0
        self.w2 = 0

    def _get_predicted_value_matrix(self, x):
        formula = lambda x: self.w0 + self.w1 * x + self.w2 * (x**2)
        return formula(x)

    def _get_gradient_matrix(self):
        gradient_matrix = [
            sum((self.y_vector - self._get_predicted_value_matrix(self.x_vector))),
            sum(
                (self.y_vector - self._get_predicted_value_matrix(self.x_vector))
                * self.x_vector
            ),
            sum(
                (self.y_vector - self._get_predicted_value_matrix(self.x_vector))
                * (self.x_vector**2)
            ),
        ]

        gradient_matrix = np.array(gradient_matrix) * -2

        return gradient_matrix

    def fit(self, step_size=0.0001, num_iterations=100000):
        for i in range(1, num_iterations):
            gradient_matrix = self._get_gradient_matrix()
            self.w0 -= step_size * gradient_matrix[0]
            self.w1 -= step_size * gradient_matrix[1]
            self.w2 -= step_size * gradient_matrix[2]

    def _show_coeffiecients(self):
        print(f"w0: {self.w0}\tw1: {self.w1}\tw2: {self.w2}\t")

    def predict(self, x):
        y = self.w0 + self.w1 * x + self.w2 * x * x
        return y


x = [-2, -1, 0, 1, 2, 3]
y = [15, 12, 7, 6, 9, 16]
model = PolinomialRegression(x, y)
model.fit(num_iterations=1000000)

plt.plot(x, y)
plt.plot(x, model.predict(np.array(x)))

plt.show()
