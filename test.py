import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# Assumption : Since there's a possible seasonality and complex trend, assumed that the polynomial function is in the form
# y=w0 + w1.x + w2.x.x + w3.x.x.x + w4.sin(x) + w5.cos(x)


def weighted_moving_average(a, window=3):
    weights = [(1 / window) for _ in range(1, window + 1)]
    return np.convolve(a, weights, mode="valid")


class PolynomialRegression:
    def __init__(self, x_vector, y_vector):
        self.x_vector = np.array(x_vector)
        self.y_vector = np.array(y_vector)

        self.w0 = 1
        self.w1 = 1
        self.w2 = 1
        self.w3 = 1
        self.w4 = 1
        self.w5 = 1

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
            sum(
                (self.y_vector - self._get_predicted_value_matrix(self.x_vector))
                * (self.x_vector**3)
            ),
            sum(
                (self.y_vector - self._get_predicted_value_matrix(self.x_vector))
                * np.cos(self.x_vector)
            ),
            sum(
                (
                    self.y_vector
                    - self._get_predicted_value_matrix(self.x_vector)
                    * np.sin(self.x_vector)
                    * (-1)
                )
            ),
        ]

        gradient_matrix = np.array(gradient_matrix) * -2

        return gradient_matrix

    def fit(self, step_size=0.0001, num_iterations=1000):
        for i in range(1, num_iterations):
            step_size /= math.sqrt(i)
            gradient_matrix = self._get_gradient_matrix()
            self.w0 += step_size * gradient_matrix[0]
            self.w1 += step_size * gradient_matrix[1]
            self.w2 += step_size * gradient_matrix[2]
            self.w3 += step_size * gradient_matrix[3]
            self.w4 += step_size * gradient_matrix[4]
            self.w5 += step_size * gradient_matrix[5]

    def _show_coeffiecients(self):
        print(
            f"w0: {self.w0}\nw1: {self.w1}\nw2: {self.w2}\nw3: {self.w3}\nw4: {self.w4}\nw5: {self.w5}\n"
        )

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


x = [
    1,
    2,
    3,
]
y = [2, 3, 2]
model = PolynomialRegression(x, y)
model.fit(num_iterations=10000)


plt.plot(x, y)
plt.plot(x, model.predict(np.array(x)))

plt.show()
