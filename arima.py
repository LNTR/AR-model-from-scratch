from Autoregression import AR
from MovingAverage import MA
from acf import AutoCorrelationCalculator
import matplotlib.pyplot as plt
import numpy as np

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
    537,
    499,
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


acf = AutoCorrelationCalculator(y, lag_size=12)
acf.calculate_acf(12)
acf_values = acf.get_max_acf_values(3)
periods = acf.get_max_acf_indices(3) + 1

ar_model = AR(y, periods)
ar_model.fit(40, epoch_limit=50000)
ar_coefficients = ar_model.get_coefficients()
ar_predictions = ar_model.predicted_values
ma_model = MA(
    y,
    periods=periods,
    acf_values=acf_values,
    ar_coefficients=ar_coefficients,
    ar_weight=1,
    ma_weight=0.1325,
)

ma_model.update_arma_predictions()
ma_predictions = ma_model.get_arma_predictions()

y = np.array(y)
plt.plot(ar_predictions, label="AR Predicted")
plt.plot(ma_predictions[ar_model.index_vector], label="MA Predicted")
plt.plot(y[ar_model.index_vector], label="Actual")
plt.legend()
plt.show()
