from Autoregression import AR
from acf import AutoCorrelationCalculator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

period_length = 10
lag_size = 24

df = pd.read_csv("./inputs/monthly-beer-production-in-austr.csv")
y = df["Values"].dropna().to_list()


test_y = y[:-period_length]

acf = AutoCorrelationCalculator(test_y, lag_size)
acf.calculate_acf(24)
acf_values = acf.get_max_acf_values(period_length)
periods = acf.get_max_acf_indices(period_length) + 1

ar_model = AR(test_y, periods)
ar_model.fit(0.001, epoch_limit=10000)

ar_coefficients = ar_model.get_coefficients()
ar_predictions = ar_model.predict_next_batch()

print(ar_predictions[-period_length:])
print(y[-period_length:])

y = np.array(y)

current_y = y[ar_model.index_vector]
future_y = y[-period_length:]
x = len(current_y)

future_x = np.array(range(x, x + period_length))

plt.plot(ar_predictions, label="AR Predicted")
plt.plot(current_y, label="Actual")
plt.plot(future_x, future_y, label="Actual Future", color="red")

plt.legend()
plt.show()
