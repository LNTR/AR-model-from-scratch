# Import the libraries
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

n = 100
e = np.random.normal(size=n)
y = np.zeros(n)
y[0] = 10 + e[0]
y[1] = 10 + e[1] - 0.6 * e[0]
for t in range(2, n):
    y[t] = 10 + e[t] - 0.6 * e[t - 1] + 0.4 * e[t - 2]

model = sm.tsa.ARIMA(y, order=(0, 0, 2))
result = model.fit()

print(result.summary())

result.plot_diagnostics()
plt.show()

pred = result.predict(start=n, end=n + 9)
print(pred)
