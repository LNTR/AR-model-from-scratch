def calculate_correlation(var1, var2):

    if len(var1) != len(var2):
        raise ValueError("Both variables should have the same number of data points.")
    correlation_matrix = np.corrcoef(var1, var2)
    correlation = correlation_matrix[0, 1]
    return correlation


collerations = np.array([])
lag_size = 6
data_length = len(data)

for i in range(1, 5):
    collerations = np.append(
        collerations,
        abs(
            calculate_correlation(
                data[data_length - lag_size - i : data_length - i],
                data[data_length - lag_size - i - 1 : data_length - i - 1],
            )
        ),
    )


collerations = np.sort(collerations)[::-1]
print(collerations)
