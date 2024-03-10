# used for cleaning and transforming the inputs inside the excel
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import numpy as np
import math


def weighted_moving_average(array, window=5):
    weights = [(1 / window) for _ in range(1, window + 1)]
    offset_count = math.floor(window / 2)
    array = np.insert(array, 0, [array[0] for _ in range(offset_count)])
    array = np.insert(array, 0, [array[-1] for _ in range(offset_count)])
    if not (window % 2):
        array = np.delete(array, [-1])
    return np.convolve(array, weights, mode="valid")


def calculate_ema(data, alpha):
    data = np.array(data)
    ema = [data[0]]
    for i in range(1, len(data)):
        ema_temp = alpha * ema[i - 1] + (1 - alpha) * data[i]
        ema.append(ema_temp)
    return ema


df = pd.read_excel("cleaned_inputs.xlsx")

# without_noise = weighted_moving_average(df["400g Crystal"])
# plt.plot(df.index, without_noise)

# residuals = df["400g Crystal"].to_numpy() - without_noise

date_to_int = {date: i for i, date in enumerate(df.index.unique())}
df["date_code"] = df.index.to_series().map(date_to_int)


x = list(range(0, len(df["date_code"].to_list())))
# y = without_noise.tolist()
# y_pred = calculate_ema(df["400g Crystal"].to_list(), 0.5)


# alpha = find_optimal_alpha(x, y)
plt.plot(x, np.log10(df["400g Crystal"].to_numpy()))

# plt.plot(x, y_pred)

plt.show()
