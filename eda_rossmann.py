#%%
# 1 - Libs Imports
import pandas as pd
import inflection

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
cols_new_names

# %%

# 4 - Data Description
print(f'The number of rows is: {df.shape[0]}')
print(f'The number of columns is: {df.shape[1]}')


# %%



# %%
