import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings('ignore')
sns.set(style='white')


def outlier_show(variable):
    sns.boxplot(df[variable])
    plt.show()


# Extract the index of the row with outliers from the feature
def outlier(df, feature):
    out_indexer = []
    for i in feature:
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)

        IQR = Q3 - Q1

        alt_sinir = Q1 - 1.5 * IQR
        ust_sinir = Q3 + 1.5 * IQR

        out = ((df[i] < alt_sinir) | (df[i] > ust_sinir))

        out_index = df[i][out].index
        out_indexer.extend(out_index)

    out_indexer = Counter(out_indexer)

    outlier_index = [i for i, v in out_indexer.items() if v > 0]
    return outlier_index


df = pd.read_csv('Banking_churn_prediction.csv')

# Convert churn, branch_code, customer_nw_category to category type
df['churn'] = df['churn'].astype('category')
df['branch_code'] = df['branch_code'].astype('category')
df['customer_nw_category'] = df['customer_nw_category'].astype('category')

# Change dependents that values are over 4 into value 3
fil = (df['dependents'] == 4) | (df['dependents'] == 5)\
      | (df['dependents'] == 6) | (df['dependents'] == 7)\
      | (df['dependents'] == 32) | (df['dependents'] == 50)\
      | (df['dependents'] == 36) | (df['dependents'] == 52)\
      | (df['dependents'] == 8) | (df['dependents'] == 9)\
      | (df['dependents'] == 25)
df.loc[fil, 'dependents'] = 3

# Convert dependents, city to category type
df['dependents'] = df['dependents'].astype('category')
df['city'] = df['city'].astype('category')

# Change gender and occupation type into category
df['gender'] = df['gender'].astype('category')
df['occupation'] = df['occupation'].astype('category')

# Creating an instance(data) of Datetimeindex class using last_transaction
df['last_transaction'] = pd.DatetimeIndex(df['last_transaction'])

# extract new columns from last_transaction
date = pd.DatetimeIndex(df['last_transaction'])

df['last_transaction'].drop()
df['last_transaction'] = date.month
print(df['last_transaction'].value_counts())

# Drop customer_id
df.drop(['customer_id'], axis=1, inplace=True)

# Dealing with nan values in gender
dict_gender = {'Male': 1, 'Female': 0}
df.replace({'gender': dict_gender}, inplace=True)
df['gender'] = df['gender'].fillna(method='ffill')

# Dealing with nan values in dependents
df['dependents'] = df['dependents'].fillna(0.0)

# Dealing with nan values in occupation
df['occupation'] = df['occupation'].fillna('self_employed')

# Dealing with nan values in city
df['city'] = df['city'].fillna(1020)

# Now change gender into category type
df['gender'] = df['gender'].astype('category')

# Show outlier value.
outlier_list = ['vintage', 'age']
for i in outlier_list:
    outlier_show(i)

# Find outlier indexes
outlier_index = outlier(df, ['vintage', 'age'])

# Deletion outlier values.
df = df.drop(outlier_index, axis=0).reset_index(drop=True)
print(df)