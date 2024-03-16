# import pandas as pd

# df = pd.read_excel("cleaned_inputs2.xlsx")

# df["Date (mm-dd-yyyy)"] = pd.to_datetime(df["Date (mm-dd-yyyy)"])
# df.set_index("Date (mm-dd-yyyy)", inplace=True)

# monthly_data = df.groupby(pd.Grouper(freq="M")).sum()
# print(monthly_data)


import re


def extract_patterns(text):
    pattern = r"\d+"  # I want to extract all sequences of digits
    matches = re.findall(pattern, text)
    if matches:
        return matches
    else:
        return "No match found"


text = "123 apples, 456 oranges, and 789 bananas"
result = extract_patterns(text)
print(result)
