# used for cleaning and transforming the inputs inside the excel
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("inputs.xlsx")
df.dropna(axis=0, subset=["Date (mm-dd-yyyy)"], how="any", inplace=True)
df["Date (mm-dd-yyyy)"] = pd.to_datetime(df["Date (mm-dd-yyyy)"]).dt.strftime(
    "%Y-%m-%d"
)
cleand_df = df.groupby("Date (mm-dd-yyyy)").sum()
cleand_df.to_excel("cleaned_inputs.xlsx")
cleand_df["400g Crystal"]
plt.plot(cleand_df.index, cleand_df["400g Crystal"].to_list())

plt.show()
# print()
