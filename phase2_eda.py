import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("home-credit-default-risk/application_train.csv")

# ------------------------------------
# TARGET DISTRIBUTION
# ------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(x='TARGET', data=df)

plt.title("Loan Repayment Status")
plt.xlabel("Target")
plt.ylabel("Count")

plt.show()

# ------------------------------------
# INCOME DISTRIBUTION
# ------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    df['AMT_INCOME_TOTAL'],
    bins=50,
    kde=True
)

plt.title("Income Distribution")

plt.show()

# ------------------------------------
# LOAN AMOUNT DISTRIBUTION
# ------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    df['AMT_CREDIT'],
    bins=50,
    kde=True
)

plt.title("Credit Amount Distribution")

plt.show()

# ------------------------------------
# TARGET VS GENDER
# ------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(
    x='CODE_GENDER',
    hue='TARGET',
    data=df
)

plt.title("Default by Gender")

plt.show()

# ------------------------------------
# TARGET VS CONTRACT TYPE
# ------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(
    x='NAME_CONTRACT_TYPE',
    hue='TARGET',
    data=df
)

plt.title("Default by Contract Type")

plt.show()
print("\nDEFAULT RATE")

default_rate = (
    df.groupby('TARGET')
      .size()
      / len(df)
)

print(default_rate)