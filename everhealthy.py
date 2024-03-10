import matplotlib.pyplot as plt

# Assuming you have your data and predictions in the following lists:
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # replace with your data
predictions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # replace with your predictions

# Split data into real and predicted parts
real_data = data[:-10]
predicted_data = data[-10:]

# Create a combined list for x-axis
x = list(range(len(real_data) + len(predictions)))

# Plot real data
plt.plot(x[: len(real_data)], real_data, label="Real")

# Plot predicted data
plt.plot(x[len(real_data) :], predicted_data, label="Predicted", linestyle="--")

# Plot predictions
plt.plot(x[len(real_data) :], predictions, label="Model Predictions", linestyle=":")

plt.legend()
plt.show()
