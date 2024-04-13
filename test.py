import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AR:
    def __init__(self, y_vector):
        self.start_index = 2
        self.y_vector = np.array(y_vector, dtype=np.float64)
        self.error_vector = np.array([0, 0], dtype=np.float64)

        self.index_vector = np.arange(self.start_index, len(self.y_vector))

        self.c = 0
        self.r1 = 0
        self.r2 = 0
        self.r3 = 0

        self.epsilon = 1

        self.m = 0.25
        self.v = 0.25

        self.momentum = np.zeros(4)
        self.velocity = np.zeros(4)

    def _get_predicted_value(self, index):
        y = self.r1 * self.y_vector[index - 1] + self.r1 * self.y_vector[index - 1]
        return y

    def _get_gradient_matrix(self):
        predicted_value_vector = []
        for index in self.index_vector:
            predicted_value_vector.append(self._get_predicted_value(index))

        gradient_matrix = [
            sum((self.y_vector[self.start_index :] - predicted_value_vector)),
            sum(
                (self.y_vector[self.start_index :] - predicted_value_vector)
                * self.y_vector[self.start_index - 1 : len(self.y_vector) - 1]
            ),
            sum(
                (self.y_vector[self.start_index :] - predicted_value_vector)
                * (self.y_vector[self.start_index - 2 : len(self.y_vector) - 2])
            ),
        ]

        gradient_matrix = np.array(gradient_matrix, dtype=np.float64) * -2

        return gradient_matrix

    def fit(self, step_size=0.001, num_iterations=100000):
        self.step_size = step_size
        self.error_vector = np.array([0, 0], dtype=np.float64)

        for _ in range(1, num_iterations):
            self._update_momentum()
            self._update_velocity()
            self.c -= (
                self.step_size
                * self.momentum[0]
                / (self.epsilon + np.sqrt(self.velocity[0]))
            )
            self.r1 -= (
                self.step_size
                * self.momentum[1]
                / (self.epsilon + np.sqrt(self.velocity[1]))
            )
            self.r2 -= (
                self.step_size
                * self.momentum[2]
                / (self.epsilon + np.sqrt(self.velocity[2]))
            )

    def _show_coeffiecients(self):
        print(f"c: {self.c}\tr1: {self.r1}\tr2: {self.r2}\t\t")

    def view_inside(self):
        y_vals = self._get_predicted_value(self.index_vector)
        return y_vals

    def _get_current_weigts(self):
        return np.array([self.c, self.r1, self.r2])

    def _update_momentum(self):
        gradient_matrix = self._get_gradient_matrix()
        self.momentum = self.v * self.momentum + (1 - self.v) * gradient_matrix

    def _update_velocity(self):
        gradient_matrix = self._get_gradient_matrix()
        self.velocity = self.v * self.velocity + (1 - self.v) * (gradient_matrix**2)


class MA:
    pass


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

model = AR(np.log10(y))
model.fit(num_iterations=100000)
inside = model.view_inside()
inside = np.power(10, inside[3:])
y = np.array(y[5:])

print(y)
print(inside[3:])
plt.plot(y, label="Actual")
plt.plot(inside, label="Predict")
plt.legend()
plt.show()
