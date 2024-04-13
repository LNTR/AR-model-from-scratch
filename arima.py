import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AR:
    def __init__(self, y_vector, periods):
        self.y_vector = np.array(y_vector, dtype=np.float64)
        self.start_index = np.max(periods)
        self.index_vector = np.arange(self.start_index, len(self.y_vector))

        self.coefficients = np.zeros(len(periods))
        self.periods = np.array(periods)

        self.epsilon = 1

        self.m = 0.25
        self.v = 0.25

        self.momentum = np.zeros(len(periods))
        self.velocity = np.zeros(len(periods))
        self.predicted_value_vector = np.zeros(len(self.y_vector) - self.start_index)

    def _get_predicted_value(self, index):
        y = np.sum(self.coefficients * self.y_vector[index - self.periods])
        return y

    def _run_vectorized(self, iterator, function):
        vectorized_function = np.vectorize(function)
        return vectorized_function(iterator)

    def _get_gradient_matrix(self):

        self.predicted_value_vector = self._run_vectorized(
            self.index_vector, self._get_predicted_value
        )

        gradient_matrix = self._run_vectorized(self.periods, self._get_gradient_value)

        gradient_matrix = np.array(gradient_matrix, dtype=np.float64) * -2
        return gradient_matrix

    def _get_gradient_value(self, current_index):

        difference = self.y_vector[self.start_index :] - self.predicted_value_vector
        shifted_vector = self.y_vector[
            self.start_index - current_index : len(self.y_vector) - current_index
        ]
        return np.sum(difference * shifted_vector)

    def fit(self, acceptable_mse=50, step_size=0.0001):
        self.step_size = step_size

        while self._get_mse() > acceptable_mse:
            self._update_coefficients()

    def _get_current_weigts(self):
        return np.array([self.c, self.r1, self.r2])

    def _update_momentum(self):
        gradient_matrix = self._get_gradient_matrix()
        self.momentum = self.v * self.momentum + (1 - self.v) * gradient_matrix

    def _update_velocity(self):
        gradient_matrix = self._get_gradient_matrix()
        self.velocity = self.v * self.velocity + (1 - self.v) * (gradient_matrix**2)

    def _update_coefficients(self):
        self._update_momentum()
        self._update_velocity()
        for i in range(len(self.periods)):
            self.coefficients[i] -= (
                self.step_size
                * self.momentum[i]
                / (self.epsilon + np.sqrt(self.velocity[i]))
            )

    def print_coefficients(self):
        print(self.coefficients)

    def _get_mse(self):
        residual_sum = np.sum(
            (self.y_vector[self.start_index :] - self.predicted_value_vector) ** 2
        )
        mse = np.sqrt(residual_sum) / len(self.y_vector)
        return mse


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
    368,
    200,
    190,
    737,
    699,
    124,
    683,
    284,
    554,
    223,
    471,
    221,
    566,
    577,
    546,
    327,
    589,
]

model = AR(y, periods=(2, 3, 10))
model.fit(31)

model.print_coefficients()
# print(model._get_mse())
y = np.array(y)
predictions = 0.31019374 * y[12:-2] + 0.25943325 * y[11:-3] + 0.46238168 * y[10:-4]
plt.plot(predictions, label="Predicted")
plt.plot(y[10:], label="Actual")

plt.legend()
plt.show()
