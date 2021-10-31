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

def UVA_numeric(data, var_group):
    size = len(var_group)
    plt.figure(figsize=(7*size, 7), dpi=100)

    for j, i in enumerate(var_group):
        mini = data[i].min()
        maxi = data[i].max()
        ran = data[i].max()-data[i].min()
        mean = data[i].mean()
        median = data[i].median()
        st_dev = data[i].std()
        skew = data[i].skew()
        kurt = data[i].kurtosis()

        # Calculate std points
        points = mean - st_dev, mean + st_dev

        # Plot
        plt.subplot(1, size, j + 1)
        sns.kdeplot(data[i], shade=True)
        sns.lineplot(points, [0, 0], color='black', label="std_dev")
        sns.scatterplot([mini, maxi], [0, 0], color='orange', label="min/max")
        sns.scatterplot([mean], [0], color='red', label="mean")
        sns.scatterplot([median], [0], color='blue', label="median")
        plt.xlabel('{}'.format(i), fontsize=20)
        plt.ylabel('density')
        plt.title('std_dev = {}; kurtosis = {};'
                  '\nskew = {}; range = {}'
                  '\nmean = {}; median = {}'
                  ''.format((round(points[0], 2), round(points[1], 2)),
                            round(kurt, 2), round(skew, 2),
                            (round(mini, 2), round(maxi, 2), round(ran, 2)),
                            round(mean, 2), round(median, 2)))
    plt.show()


def UVA_category(data, var_group):
    size = len(var_group)
    plt.figure(figsize=(7 * size, 7), dpi=100)

    for j, i in enumerate(var_group):
        norm_count = data[i].value_counts(normalize=True)
        n_uni = data[i].nunique()

        plt.subplot(1, size, j + 1)
        sns.barplot(norm_count, norm_count.index, order=norm_count.index)
        plt.xlabel('fraction/percent', fontsize=20)
        plt.ylabel('{}'.format(i), fontsize=20)
        plt.title('n_uniques = {}'
                  '\n value counts \n {};'.format(n_uni, norm_count))

    plt.show()


df = pd.read_csv('Banking_churn_prediction.csv')

# print(str(df.shape) + "\n")
# print(str(df.shape) + "\n")

# Print all the columns in dataset
# print(str(df.columns) + "\n")

# Print the data types of different variables (columns)
# print(str(df.dtypes) + "\n")

# Print the list of columns with data type = int64
# print(str(df.dtypes[df.dtypes == 'int64']) + "\n")
# Churn, branch_code, customer_nw_category should be converted to "Category"

# Convert churn, branch_code, customer_nw_category to category type
df['churn'] = df['churn'].astype('category')
df['branch_code'] = df['branch_code'].astype('category')
df['customer_nw_category'] = df['customer_nw_category'].astype('category')

# We can see churn, branch_code, customer_nw_category disappear
# print(str(df.dtypes[df.dtypes == 'int64']) + "\n")

# Print the list of columns with data type = float64
# print(str(df.dtypes[df.dtypes == 'float64']) + "\n")
# We need to check the frequency of dependent
# City and dependent should be converted to "Category"

# Print the frequency of dependents
# print(str(df['dependents'].value_counts(normalize=True)) + "\n")
# Frequency of over 4 is too low, so integrate them into 3

# Change dependents that values are over 4 into value 3
fil = (df['dependents'] == 4) | (df['dependents'] == 5)\
      | (df['dependents'] == 6) | (df['dependents'] == 7)\
      | (df['dependents'] == 32) | (df['dependents'] == 50)\
      | (df['dependents'] == 36) | (df['dependents'] == 52)\
      | (df['dependents'] == 8) | (df['dependents'] == 9)\
      | (df['dependents'] == 25)
df.loc[fil, 'dependents'] = 3

# Print the frequency of dependent that have only values 0, 1, 2, 3
# print(str(df['dependents'].value_counts(normalize=True)) + "\n")

# Convert dependents, city to category type
df['dependents'] = df['dependents'].astype('category')
df['city'] = df['city'].astype('category')

# Check city and dependents type
# print(str(df[['city', 'dependents']].dtypes) + "\n")
# We can check city and dependents type changed into category well

# Print the whole types of all columns
# print(str(df.dtypes) + "\n")
# columns that's type is object is gender, occupation and last_transaction
# Print the head of 3 columns
# print(str(df[['gender', 'occupation', 'last_transaction']].head(10)) + "\n")
# gender and occupation columns belongs to category
# last_transaction should be datetime variable

# Change gender and occupation type into category
df['gender'] = df['gender'].astype('category')
df['occupation'] = df['occupation'].astype('category')

# Check gender and occupation type
# print(str(df[['gender', 'occupation']].dtypes) + "\n")
# We can check gender and occupation type changed into category well

# Creating an instance(data) of Datetimeindex class using last_transaction
df['last_transaction'] = pd.DatetimeIndex(df['last_transaction'])

# extract new columns from last_transaction
date = pd.DatetimeIndex(df['last_transaction'])

# ls_tran = last_transaction
# doy = date of year
df['doy_ls_tran'] = date.dayofyear
# woy = week of year
df['woy_ls_tran'] = date.weekofyear
# moy = month -> month of year (to match 3 words like others)
df['moy_ls_tran'] = date.month
# dow = day of week
# Monday 0 ~ Sunday 6 (즉 요일이다)
df['dow_ls_tran'] = date.dayofweek

# Print new column created from last_transaction column and last_transaction
"""
print(str(df[['last_transaction', 'doy_ls_tran',
              'woy_ls_tran', 'moy_ls_tran',
              'dow_ls_tran']].head(10)) + "\n")
"""
# Finishing Exploratory Dataset

# UNIVARIATE ANALYSIS (일원분석 일변량분석이라고도 함)
# 단일 변수를 묘사하는 목적

# Print the columns that types are int64 or float64
# print(str(df.select_dtypes(include=['int64', 'float64']).dtypes) + "\n")

# Separate the columns from types int64 or float64
customer_details = ['customer_id', 'vintage', 'age']

current_month = ['current_balance', 'current_month_credit',
                 'current_month_debit', 'current_month_balance']
previous_month = ['previous_month_end_balance', 'previous_month_credit',
                  'previous_month_debit', 'previous_month_balance']
previous_quarters = ['average_monthly_balance_prevQ',
                     'average_monthly_balance_prevQ2']
transaction_date = ['doy_ls_tran', 'woy_ls_tran',
                    'moy_ls_tran', 'dow_ls_tran']


# Univariate Analysis of customer_details
# UVA_numeric(df, customer_details)
# Result shows
# For customer_id -> every customer have unique values -> no helpful to use
# So we will drop customer_id

# For age -> median age is 46, most customers age is between 30~66
# skewness(비대칭도) is +0.33, which means negligibly biased towards younger age
# kurtosis(첨도: 확률분포의 꼬리의 두꺼운 정도)
# is -0.17, which means very less likely to have extreme/outlier values.

# For vintage -> most customers joined between 2100~2650 days
# skewness is -1.42 which means left skewed,
# vintage variable is significantly biased towards longer association of customers
# Kurtosis is 2.93 which means there are outliers in vintage

# Drop customer_id
df.drop(['customer_id'], axis=1, inplace=True)

# Univariate Analysis of current_month
# UVA_numeric(df, current_month)
# Considering kurtosis and skewness of result, there are outliers

# Univariate Analysis of previous_month
# UVA_numeric(df, previous_month)
# Result is very similar with current_month
# -> there are relationship between previous and current month data

# Univariate Analysis of transaction data
# UVA_numeric(df, transaction_date)
# For doy(day of year)
# most of the last transactions
# were made in the last 60 days of the extraction of data.
# There are transactions which were made more than an year ago.

# For woy(week of year) and moy(month of year)
# These variable validate the findings from the doy

# dow(day of week) shows there are no transaction in sunday
# And high number of transaction on monday

# Univariate Analysis of Categorical variables
# UVA_category(df, ['occupation', 'gender', 'customer_nw_category', 'dependents'])
# For occupation
# -> 60% of people are self_employed, 1% company accounts

# For gender
# -> males accounts are 1.5 times in number of females (male:60% female:40%)

# For customer_nw_category
# -> Half of all the accounts belong to the 3rd net worth category.
# Less than 15% belong to the highest net worth category.

# For dependents
# -> Majority of the population has 0 dependents.
# Which means they have no one to take care off on their expenses.

# UVA_category(df, ['churn'])
# For churn
# -> Number of people who churned are 1/4 times of the people
# who did not churn in the given data.

# Analysis of missing values
# print(str(df.isnull().sum()) + "\n")
# gender, dependents, occupation, city, last_transaction(split into 4 columns)

# Before dealing with nan values in gender
# print(str(df['gender'].value_counts()) + "\n")
dict_gender = {'Male': 1, 'Female': 0}
df.replace({'gender': dict_gender}, inplace=True)
df['gender'] = df['gender'].fillna(method='ffill')
# After dealing with nan values in gender
# print(str(df['gender'].value_counts()) + "\n")

# Before dealing with nan values in dependents
# print(str(df['dependents'].value_counts()) + "\n")
# Because most of values are 0.0  fillna with 0
df['dependents'] = df['dependents'].fillna(0.0)
# After dealing with nan values in dependents
# print(str(df['dependents'].value_counts()) + "\n")

# Before dealing with nan values in occupation
# print(str(df['occupation'].value_counts()) + "\n")
# Because most of values are self_employed fillna with 'self_employed'
df['occupation'] = df['occupation'].fillna('self_employed')
# After dealing with nan values in occupation
# print(str(df['occupation'].value_counts()) + "\n")

# Before dealing with nan values in city
# print(str(df['city'].value_counts()) + "\n")
# It looks hard so calculate mode of city
# print(str(df['city'].mode()) + "\n")
# 1020 is the mode value of city so fillna with 1020
df['city'] = df['city'].fillna(1020)
# After dealing with nan values in city
# print(str(df['city'].value_counts()) + "\n")

# Now change gender into category type
df['gender'] = df['gender'].astype('category')

print(str(df.isnull().sum()) + "\n")

# Show outlier value.
outlier_list = ['vintage','age','dependents','current_balance','previous_month_end_balance',
'average_monthly_balance_prevQ','average_monthly_balance_prevQ2','current_month_credit',
'previous_month_credit','current_month_debit','previous_month_debit','current_month_balance',
'previous_month_balance']
for i in outlier_list:
    outlier_show(i)

# Find outlier indexes
outlier_index = outlier(df, ['vintage','age','dependents','current_balance','previous_month_end_balance',
'average_monthly_balance_prevQ','average_monthly_balance_prevQ2','current_month_credit',
'previous_month_credit','current_month_debit','previous_month_debit','current_month_balance',
'previous_month_balance'])

# Deletion outlier values.
df = df.drop(outlier_index, axis=0).reset_index(drop=True)
print(df)