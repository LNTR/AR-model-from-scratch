import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class PolynomialRegression:
    def __init__(self, y_vector):
        self.y_vector = np.array(y_vector, dtype=np.float64)
        self.error_list = np.array([0, 0])
        self.start_index = 2

        self.w0 = 0
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        self.w4 = 0

        self.epsilon = 1

        self.m = 0.25
        self.v = 0.25

        self.momentum = np.zeros(5)
        self.velocity = np.zeros(5)

    def _get_predicted_value_matrix(self, index):
        ar = self.w1 * self.y_vector[index - 1] + self.w2 * self.y_vector[index - 2]
        ma = self.w3 * self.error_list[index - 1] + self.w4 * self.error_list[index - 2]
        y = self.w0 + ar + ma
        return y

    def _get_current_weigts(self):
        return np.array([self.w0, self.w1, self.w2, self.w3])

    def _update_momentum(self):
        gradient_matrix = self._get_gradient_matrix()
        self.momentum = self.v * self.momentum + (1 - self.v) * gradient_matrix

    def _update_velocity(self):
        gradient_matrix = self._get_gradient_matrix()
        self.velocity = self.v * self.velocity + (1 - self.v) * (gradient_matrix**2)

    def _get_gradient_matrix(self):
        predicted_value_vector = self._get_predicted_value_matrix(self.x_vector)
        gradient_matrix = [
            sum((self.y_vector - predicted_value_vector)),
            sum(
                (self.y_vector - predicted_value_vector)
                * self.y_vector[self.start_index - 1 :]
            ),
            sum(
                (self.y_vector - predicted_value_vector)
                * (self.y_vector[self.start_index - 2 :])
            ),
            sum(
                (self.y_vector - predicted_value_vector)
                * (self.error_list[self.start_index - 1 :])
            ),
            sum(
                (self.y_vector - predicted_value_vector)
                * (self.error_list[self.start_index - 2 :])
            ),
        ]

        gradient_matrix = np.array(gradient_matrix, dtype=np.float64) * -2

        return gradient_matrix

    def fit(self, step_size=1, num_iterations=100000):
        self.step_size = step_size
        for _ in range(1, num_iterations):
            self._update_momentum()
            self._update_velocity()
            self.w0 -= (
                self.step_size
                * self.momentum[0]
                / (self.epsilon + np.sqrt(self.velocity[0]))
            )
            self.w1 -= (
                self.step_size
                * self.momentum[1]
                / (self.epsilon + np.sqrt(self.velocity[1]))
            )
            self.w2 -= (
                self.step_size
                * self.momentum[2]
                / (self.epsilon + np.sqrt(self.velocity[2]))
            )
            self.w3 -= (
                self.step_size
                * self.momentum[3]
                / (self.epsilon + np.sqrt(self.velocity[3]))
            )
            self.w3 -= (
                self.step_size
                * self.momentum[4]
                / (self.epsilon + np.sqrt(self.velocity[4]))
            )

    def _show_coeffiecients(self):
        print(f"w0: {self.w0}\tw1: {self.w1}\tw2: {self.w2}\tw3: {self.w3}\t")

    def predict(self, x):
        y = self.w0 + self.w1 * x + self.w2 * (x**2) + self.w3 * (x**3)
        return y


y = [
    451,
    326,
    319,
    398,
    427,
    396,
    309,
    483,
    162,
    600,
    190,
    737,
    699,
    124,
]
x = list(range(1, len(y) + 1))

model = PolynomialRegression(x, y)
model.fit(num_iterations=100000)

plt.plot(x, y, label="Actual")
plt.plot(x, model.predict(np.array(x)), label="Predict")
# plt.legend()
# model._show_coeffiecients()
plt.show()
# np.set_printoptions(suppress=True, precision=3)

# pred = model.predict(np.array([1000, -1500, -9635, 9635], dtype=np.float64))
# test = np.array([f(x_val) for x_val in [1000, -1500, -9635, 9635]], dtype=np.float64)
# accuracy = 1 - (sum((test - pred) ** 2) / sum((test - test.mean()) ** 2))
# print(np.array_str(pred, precision=3, suppress_small=True))
# print(np.array_str(test, precision=3, suppress_small=True))
# print(accuracy)
