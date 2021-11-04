import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv("C:/Users/MOS/Desktop/Banking_churn_prediction.csv")

#Churn plot-------
Churn_plot = df['churn'].value_counts().reset_index()
Churn_plot.columns = ['churn', 'count']
plt.figure(figsize=(7, 7))
sns.countplot(x=df['churn'])
plt.title('Rate of churn', fontsize=20)
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show() 
#------------------


#Occupation plot-------
plt.figure(figsize=(7, 7))
sns.countplot(x=df['occupation'])
plt.title('Rate of occupation', fontsize=20)
plt.xlabel('occupation')
plt.ylabel('Count')
plt.show() 
#------------------

#gender plot-------
plt.figure(figsize=(7, 7))
sns.countplot(x=df['gender'])
plt.title('Rate of gender', fontsize=20)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
#------------------

# branch_code plot-------
plt.bar(range(len(df['branch_code'])), df['branch_code'])
plt.title('Rate of branch_code', fontsize=20)
plt.xlabel('branch_code')
plt.ylabel('Count')
plt.show()
# ------------------

#age plot-------
df[["age"]].hist(bins=10)
plt.title('Rate of age', fontsize=20)
plt.xlabel('age')
plt.ylabel('Count')
plt.show()
#------------------


# #find outlier---------
# fig, ax = plt.subplots(1, 4, figsize=(16, 4))
# ax[0].boxplot(df['current_balance'])
# ax[0].set_title("current_balance")
# ax[1].boxplot(df['current_month_credit'])
# ax[1].set_title("current_month_credit")
# ax[2].boxplot(df['current_month_debit'])
# ax[2].set_title("current_month_debit")
# ax[3].boxplot(df['current_month_balance'])
# ax[3].set_title("current_month_balance")
# plt.show()
# # #------------------

# #Relation about (previous & current balance/credit/debit/mothly balance) & (Churn)  
# fig = plt.figure(figsize=(7,7))
# graph = sns.scatterplot(data=df, x='previous_month_end_balance', y='current_balance', hue='churn')
# plt.show()
# graph = sns.scatterplot(data=df, x='previous_month_credit', y='current_month_credit', hue='churn')
# plt.show()
# graph = sns.scatterplot(data=df, x='previous_month_debit', y='current_month_debit', hue='churn')
# plt.show()
# graph = sns.scatterplot(data=df, x='previous_month_balance', y='current_month_balance', hue='churn')
# plt.show()


# df.dropna(axis=0, inplace=True)
# df.drop(['last_transaction'], axis=1, inplace=True)
# feature_names = list(df.select_dtypes(object))


# X = df.drop(['churn'], axis=1)
# y = df['churn']
# x = df.copy()


# encoder = LabelEncoder()
# data=pd.DataFrame()
# x=x.reset_index(drop=True)  #index가 있는채로 인코딩하면 결측값이 생기므로 인덱스를 reset해주고 drop=True를 해서 다른 column으로 나오는것을 방지함
# for i in feature_names:
#     x[i] = encoder.fit_transform(x[i])
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# df_new = pd.DataFrame(x)
# df_new.columns = df.columns

# X = df_new.drop(['churn'], axis=1)
# y = df_new['churn']

# selector = SelectKBest(score_func=f_classif, k=19)
# fit = selector.fit(X, y)

# dfcolumns = pd.DataFrame(X.columns)
# dfscores = pd.DataFrame(fit.scores_)

# featureScores = pd.concat([dfcolumns, dfscores], axis=1)
# featureScores.columns = ['Spec', 'Score']


# print(featureScores.nlargest(20, 'Score'))