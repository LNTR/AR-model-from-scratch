import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel("cleaned_inputs2.xlsx")


class AR:

    def __init__(self, data, alpha=0.5, window_size=20):
        self.alpha = alpha
        self.AR_list = [0 for _ in range(window_size)]
        self.window_size = window_size
        self.data = data

    def update(self, value):
        t = self.alpha * value
        window_ar_list = []
        for i in range(1, self.window_size + 1):
            current_ar = self.AR_list[i * (-1)] * ((1 - self.alpha) ** (i + 1))
            window_ar_list.append(current_ar)

        current_ar = self.AR_list[0] * ((1 - self.alpha) ** (i + 1))

        new_ar = t + sum(window_ar_list)
        self.AR_list.pop(0)
        self.AR_list.append(new_ar)
        return new_ar

    def predict(self, future=8):
        temp_list = self.AR_list
        predictions = []
        value = data[-1]

        for _ in range(future):
            t = self.alpha * value
            window_ar_list = []
            for i in range(1, self.window_size + 1):
                current_ar = temp_list[i * (-1)] * ((1 - self.alpha) ** (i + 1))
                window_ar_list.append(current_ar)

            current_ar = temp_list[0] * ((1 - self.alpha) ** (i + 1))
            new_ar = t + sum(window_ar_list)
            value = temp_list.pop(0)
            predictions.append(new_ar)
            temp_list.append(new_ar)
        return predictions


data = df["400g Crystal"].to_list()[:-20]

ar_model = AR(alpha=0.75, data=data)
ar_list = []

for value in data:
    ar_list.append(ar_model.update(value))

predictions = ar_model.predict(20)
plt.plot(df["400g Crystal"].to_list())

plt.plot(ar_list + predictions)

plt.show()
