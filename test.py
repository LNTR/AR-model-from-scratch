import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


df = pd.read_excel("cleand_groups_2.xlsx")

data = df["400g FINE"]
data = data.to_list()[:-2]
print(data)
test = df["400g FINE"].to_list()[-2:]
data = np.array(data)

print(len(data))


error_list = [0, 0]


def update_error_list(y, t):
    error = data[t] - y
    error_list.append(error)


def get_value(t, data):
    ar = data[t - 1] * 0.675 + data[t - 2] * 0.325
    return ar


value_list = [0, 0]
for t in range(2, len(data)):
    value = get_value(t, data)
    update_error_list(value, t)
    value_list.append(value)

pred = []
for t in range(0, 2):

    value = get_value(t, value_list)
    update_error_list(value, t)
    value_list.append(value)
    pred.append(value)


predicted_values = np.array(pred)
actual_values = np.array(test)
print(predicted_values)
print(actual_values)
mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
print(100 - mape)
# plt.plot(value_list)
# plt.plot(data)

# plt.show()
