import numpy as np


class AutoCorrelationCalculator:

    def __init__(self, data, lag_size=10) -> None:
        self.data = data
        self.lag_size = lag_size
        self.collerations = np.array([])

    def _calculate_correlation(self, var1, var2):

        if len(var1) != len(var2):
            raise ValueError(
                "Both variables should have the same number of data points."
            )
        correlation_matrix = np.corrcoef(var1, var2)

        correlation = correlation_matrix[0, 1]

        return correlation

    def calculate_acf(self, t):
        data_length = len(self.data)

        for i in range(1, t + 1):
            self.collerations = np.append(
                self.collerations,
                self._calculate_correlation(
                    self.data[data_length - self.lag_size - i : data_length - i],
                    self.data[data_length - self.lag_size : data_length],
                ),
            )

    def get_acf_vector(self):
        return self.collerations

    def get_max_acf_indices(self, n):
        n *= -1
        last_n_sorted = np.argpartition(np.abs(self.collerations), n)
        return last_n_sorted[n:]

    def get_max_acf_values(self, n):
        indices = self.get_max_acf_indices(n)
        return self.collerations[indices]
