import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class PolynomialRegression:
    def __init__(self, x_vector, y_vector):
        self.x_vector = np.array(x_vector)
        self.y_vector = np.array(y_vector)

        self.w0 = 0
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        self.epsilon = 1

        self.m = 0.5
        self.v = 0.5

        self.momentum = np.zeros(4)
        self.velocity = np.zeros(4)

    def _get_predicted_value_matrix(self, x):
        formula = lambda x: self.w0 + self.w1 * x + self.w2 * (x**2) + self.w3 * (x**3)
        return formula(x)

    def _get_current_weigts(self):
        return np.array([self.w0, self.w1, self.w2, self.w3])

    def _update_momentum(self):
        gradient_matrix = self._get_gradient_matrix()
        self.momentum = self.v * self.momentum + (1 - self.v) * gradient_matrix

    def _update_velocity(self):
        gradient_matrix = self._get_gradient_matrix()
        self.velocity = self.v * self.velocity + (1 - self.v) * (gradient_matrix**2)

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
        ]

        gradient_matrix = np.array(gradient_matrix, dtype=np.float64) * -2

        return gradient_matrix

    def fit(self, step_size=0.01, num_iterations=100000):
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

    def _show_coeffiecients(self):
        print(f"w0: {self.w0}\tw1: {self.w1}\tw2: {self.w2}\tw3: {self.w3}\t")

    def predict(self, x):
        y = self.w0 + self.w1 * x + self.w2 * (x**2) + self.w3 * (x**3)
        return y


x = list(range(-50, 50))
f = lambda x: 3 * x + 7
y = [f(x_val) for x_val in x]
model = PolynomialRegression(x, y)
model.fit(num_iterations=10000)

plt.plot(x, y, label="Actual")
plt.plot(x, model.predict(np.array(x)), label="Predict")
plt.legend()
model._show_coeffiecients()
plt.show()
