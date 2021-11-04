import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score

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

# There are only 3 values that last_transaction in 2018
# All other records are in 2019 so drop 2018 values
date = pd.DatetimeIndex(df['last_transaction'])
df.drop(['last_transaction'], axis=1)
indexNames = df[date.year == 2018].index
df.drop(indexNames, inplace=True)
df = df.reset_index(drop=True)

# Dealing with nan values in last_transaction
date = pd.DatetimeIndex(df['last_transaction'])
df['last_transaction'] = date.month
last_transaction_mode = df['last_transaction'].mode()
df['last_transaction'] = df['last_transaction'].fillna(float(last_transaction_mode))

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
city_mode = df['city'].mode()
df['city'] = df['city'].fillna(float(city_mode))

print(str(df.isnull().sum()) + "\n")

# Now change gender into category type
df['gender'] = df['gender'].astype('category')

non_outlier=['dependents',
             'gender',
             'occupation',
             'city',
             'customer_nw_category',
             'branch_code',
             'churn',
             'last_transaction']

# Show outlier value.
outlier_list = df.columns.tolist()
outlier_list = [x for x in outlier_list if x not in non_outlier]
for i in outlier_list:
    print(i)
    #outlier_show(i)

# Find outlier indexes
outlier_index = outlier(df, outlier_list)

# Deletion outlier values.
df = df.drop(outlier_index, axis=0).reset_index(drop=True)
df = df.drop(df[df['average_monthly_balance_prevQ2']<0].index, axis=0).reset_index(drop=True)

for i in outlier_list:
    print(i)
    #outlier_show(i)

col=[
    'current_balance',
    'previous_month_end_balance',
    'average_monthly_balance_prevQ',
    'average_monthly_balance_prevQ2',
    'current_month_balance',
    'previous_month_balance',
    'current_month_credit',
    'previous_month_credit',
    'current_month_debit',
    'previous_month_debit'
]

df['current credit usage']=(df[col[6]]*100)/(df[col[6]]+df[col[8]])
df['previous credit usage']=(df[col[7]]*100)/(df[col[7]]+df[col[9]])
df['current total spending']=df[col[6]]+df[col[8]]
df['previous total spending']=df[col[7]]+df[col[9]]
df['current total income']=df[col[4]]-df[col[6]]-df[col[8]]
df['previous total income']=df[col[5]]-df[col[7]]-df[col[9]]
df['current income to spending ratio']=(df[col[4]]*100)/(df[col[4]]+df[col[6]]+df[col[8]])
df['previous income to spending ratio']=(df[col[5]]*100)/(df[col[5]]+df[col[7]]+df[col[9]])


def encoder(encoder, df):
    return encoder.fit_transform(df)

def scaler(scaler, df):
    return scaler.fit_transform(df)

def findbest(best_n,pmax,pmin,jump):
    if best_n == 1:
        jump = (int)(jump / 2)
        pmax = best_n + jump + 1
    elif best_n + jump > pmax:
        pmax = best_n + jump * 2 + 1
        pmin = best_n + jump
    else:
        jump = (int)(jump / 2)
        pmax = best_n + jump + 1
        pmin = best_n - jump
    return pmax,pmin,jump

def classifications(df):
    lbl=df['churn']
    df=df.drop(['churn'],axis=1)

    le = LabelEncoder()
    df['occupation'] = encoder(le, df['occupation'])

    st = StandardScaler()
    df = scaler(st, df)

    for p in range(1, 3):
        pmin = 1
        pmax = 14
        jump = 4
        best=0
        best_n=0
        start=0
        while True:
            for neighbor in range(pmin, pmax, jump):
                model = KNeighborsClassifier(n_neighbors=neighbor, p=p)
                score = cross_val_score(model, df, lbl, cv=5)
                print("p="+str(p)+", n="+str(neighbor)+", score="+str(score.mean()))
                if start==0:
                    start=score.mean()
                if best<score.mean():
                    best=score.mean()
                    best_n=neighbor
            #if start * 1.0001 > best:
            #    break;
            if jump == 1:
                break;
            start = best
            pmax,pmin,jump=findbest(best_n,pmax,pmin,jump)

        print("p="+str(p)+", best n="+str(neighbor)+", score="+str(score.mean()))

classifications(df)