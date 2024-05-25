#%%
# 1 - Libs Imports
import datetime
import inflection
import numpy as np
import pandas as pd
import seaborn as sns 

from matplotlib import pyplot as plt

pd.options.display.float_format = '{:.2f}'.format
# %%
# 2 File Reading
path = '../eda_rossmann/data/train.csv'
path1 = '../eda_rossmann/data/store.csv'
df_sales_raw = pd.read_csv(path)
df_store_raw = pd.read_csv(path1)

#%%

# 3 Merge of Datasets, Rename Columns

# 3.1 Merge of Datasets
df = df_sales_raw.merge(df_store_raw, how='left', on='Store')

# 3.2 Rename Columns
col_old_names = df.columns
snake_case = lambda x: inflection.underscore(x)
cols_new_names = list(map(snake_case,col_old_names))
df.columns = cols_new_names



# %%

# 4 Data Description
df1 = df([df['sales'] > 0] & df['open'] != 0).copy()


# 4.1 Data Dimension 
print(f'The number of rows is: {df1.shape[0]}')
print(f'The number of columns is: {df1.shape[1]}')

# 4.2 Check and Change Data Types Before NA Treatment
df1.dtypes
df1['date']  = pd.to_datetime(df['date'])


# 4.3 Check NA
df1.isna().sum()


# 4.4 NA Treatment

df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month
                                                if pd.isna(x['competition_open_since_month'])
                                                else x['competition_open_since_month'], axis=1 )

df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year
                                                if pd.isna(x['competition_open_since_year'])
                                                else x['competition_open_since_year'], axis=1 )

df1['competition_distance'] = df1.apply(lambda x: 200000 if 
                                        pd.isna(x['competition_distance'])
                                        else x['competition_distance'], axis = 1)

df1[['promo_interval','promo2_since_year', 'promo2_since_week']]
df1['promo2_since_week'] = df1['promo2_since_week'].fillna(0)
df1['promo2_since_year'] = df1['promo2_since_year'].fillna(0)

# 4.5 Check Data Types
df1.dtypes


# 4.6 Change Data Types

df1['sales'] = df1['sales'].astype(float)
df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)



# %%

# 5 Descriptive Stats
df2 = df1.copy()


# Separate Categorical and Numerical Features
num_features = df2.select_dtypes(include=['int', 'float'])
cat_features = df2.select_dtypes(exclude=['int','float', 'datetime64'])

# Descriptive Stats for Numerical Features
num_features.describe().T


#Descriptive Stats for Categorical Features

# Count of Unique Elements for Each Feature
cat_features.apply(lambda x: x.unique().shape[0])

aux = df2[df2['state_holiday'] != 0]

plt.subplot(1,3,1)
sns.boxplot(x='state_holiday', y='sales', data=aux)

plt.subplot(1,3,2)
sns.boxplot(x='store_type', y='sales', data=aux)

plt.subplot(1,3,3)
sns.boxplot(x='assortment', y='sales', data=aux)

# The data is too disperse. The number of outliers is pretty high.

# %%

# 6 Feature Engineering
df3 = df2.copy()

# Hypothesis
# Stores with more assortment should sell more
# Stores with closer competitors should sell less
# Stores with old competitors should sell more
# Stores with long time promotions should sell more
# Stores with more consecutive promotion days should sell more
# Stores open on christmas should sell more
# Stores should sell more over the years
# Stores should sell more on the second semester
# Stores should sell more after day 10
# Stores should sell less on weekends
# Stores should sell less on student holidays
# 


# %%
df2.columns
# %%
