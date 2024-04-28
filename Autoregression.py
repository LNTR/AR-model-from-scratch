import numpy as np


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
        self.predicted_values = np.zeros(len(self.y_vector) - self.start_index)

    def _get_predicted_value(self, index):
        y = np.sum(self.coefficients * self.y_vector[index - self.periods])
        return y

    def _run_vectorized(self, iterator, function, otypes=[np.float64]):
        vectorized_function = np.vectorize(function, otypes=otypes)
        return vectorized_function(iterator)

    def _get_gradient_matrix(self):
        self.predicted_values = self._run_vectorized(
            self.index_vector, self._get_predicted_value
        )

        gradient_matrix = self._run_vectorized(self.periods, self._get_gradient_value)

        gradient_matrix = np.array(gradient_matrix, dtype=np.float64) * -2
        return gradient_matrix

    def _get_gradient_value(self, current_index):

        difference = self.y_vector[self.start_index :] - self.predicted_values
        shifted_vector = self.y_vector[
            self.start_index - current_index : len(self.y_vector) - current_index
        ]
        return np.sum(difference * shifted_vector)

    def fit(self, acceptable_rmse=60, epoch_limit=10000, step_size=0.0001):
        self.step_size = step_size
        epoch_counter = 0
        while (self._get_rmse() > acceptable_rmse) and (epoch_counter < epoch_limit):
            self._update_coefficients()
            epoch_counter += 1

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

    def get_coefficients(self):
        return self.coefficients

    def get_fitted_line(self):
        return self.predicted_values

    def _get_rmse(self):
        residuals = (self.y_vector[self.start_index :] - self.predicted_values) ** 2

        residual_sum = np.sum(residuals)
        rmse = np.sqrt(residual_sum) / len(residuals)

        return rmse

    def predict_next_batch(self):
        batch = []
        last_index = np.max(self.index_vector) + 1
        for i in range(len(self.periods)):
            index = last_index + i
            y = self._get_predicted_value(index)
            self.y_vector = np.append(
                self.y_vector,
                [
                    y,
                ],
            )
            batch.append(self.y_vector[index])

        self.predicted_values = np.append(self.predicted_values, batch)
        return self.predicted_values
