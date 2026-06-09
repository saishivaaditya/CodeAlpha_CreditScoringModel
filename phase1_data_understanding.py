import pandas as pd

# Load application data
app_train = pd.read_csv("home-credit-default-risk/application_train.csv")

print("="*50)
print("DATASET SHAPE")
print("="*50)
print(app_train.shape)

print("\n")

print("="*50)
print("FIRST 5 ROWS")
print("="*50)
print(app_train.head())

print("\n")

print("="*50)
print("COLUMN INFORMATION")
print("="*50)
print(app_train.info())

print("\n")

print("="*50)
print("MISSING VALUES")
print("="*50)

missing = app_train.isnull().sum()
missing = missing[missing > 0]

print(missing.sort_values(ascending=False).head(20))

print("\n")

print("="*50)
print("TARGET DISTRIBUTION")
print("="*50)

print(app_train['TARGET'].value_counts())

print("\n")

print(app_train['TARGET'].value_counts(normalize=True)*100)