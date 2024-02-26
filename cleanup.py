# used for cleaning and transforming the inputs inside the excel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def weighted_moving_average(a, window=3):
    weights = [(1 / window) for _ in range(1, window + 1)]
    return np.convolve(a, weights, mode="valid")


df = pd.read_excel("inputs.xlsx")
df.dropna(axis=0, subset=["Date (mm-dd-yyyy)"], how="any", inplace=True)
df["Date (mm-dd-yyyy)"] = pd.to_datetime(df["Date (mm-dd-yyyy)"]).dt.strftime(
    "%Y-%m-%d"
)
cleand_df = df.groupby("Date (mm-dd-yyyy)").sum()
cleand_df.to_excel("cleaned_inputs.xlsx")

plt.plot(cleand_df.index[1:-1], cleand_df["400g Crystal"].to_list()[1:-1])
without_noise = weighted_moving_average(cleand_df["400g Crystal"])
plt.plot(cleand_df.index[1:-1], without_noise)

plt.show()
