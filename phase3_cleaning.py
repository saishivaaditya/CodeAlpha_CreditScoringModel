import pandas as pd

df = pd.read_csv('home-credit-default-risk/application_train.csv')

# ---------------------------------
# Missing Percentage
# ---------------------------------

missing_percent = (
    df.isnull()
      .sum()
      .sort_values(ascending=False)
      / len(df)
      * 100
)

missing_df = pd.DataFrame({
    'Missing_Percentage': missing_percent
})

print(missing_df.head(30))
threshold = 70

cols_to_drop = missing_percent[
    missing_percent > threshold
].index

print("Columns to drop:", len(cols_to_drop))