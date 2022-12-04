import pandas as pd


client_df = pd.read_csv('Data/client_data.csv')
price_df = pd.read_csv('Data/price_data.csv')

merged_df = pd.merge(client_df, price_df, on='id')
sample_df = merged_df.groupby('churn').sample(frac=0.2).reset_index(drop=True)

sample_df.drop('churn', axis=1, inplace=True)
sample_df.to_csv('Data/sample_df.csv', index=False)