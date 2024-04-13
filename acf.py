import numpy as np
from matplotlib import pyplot as plt

data = [
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


acf = AutoCorrelationCalculator(data)
acf.calculate_acf(10)
print(acf.get_acf_vector())
