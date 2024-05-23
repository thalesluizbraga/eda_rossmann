#%%
# 1 - Libs Imports
import datetime
import inflection

import pandas as pd



# %%
# 2 File Reading
path = '../eda_rossmann/data/train.csv'
path1 = '../eda_rossmann/data/store.csv'
df_sales_raw = pd.read_csv(path)
df_store_raw = pd.read_csv(path1)

# 3 Merge of Datasets and Rename Columns

# 3.1 Merge of Datasets
df = df_sales_raw.merge(df_store_raw, how='left', on='Store')

#%%
# 3.2 Rename Columns
col_old_names = df.columns
snake_case = lambda x: inflection.underscore(x)
cols_new_names = list(map(snake_case,col_old_names))
df.columns = cols_new_names

# %%

# 4 Data Description
df1 = df.copy()


# 4.1 Data Dimension 
print(f'The number of rows is: {df1.shape[0]}')
print(f'The number of columns is: {df1.shape[1]}')




# 4.3 Check NA
df1.isna().sum()


# %%

# 4.4 NA Treatmet
df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month
                                                if pd.isna(x['competition_open_since_month'])
                                                else x['competition_open_since_month'], axis=1 )


# %%

# 4.2 Data Types
df1.dtypes

# 4.3 Data Types Transformation

# %%
df1.head()
# %%
df1['date']  = pd.to_datetime(df['date'])
df1['sales'] = df1['sales'].astype(float)
df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)

# %%
df1.dtypes
# %%
