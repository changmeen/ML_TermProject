import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv("C:/Users/MOS/Desktop/Banking_churn_prediction.csv")

# def numerical_vis_func(x):
#     plt.hist(x,bins=10)
#     plt.show()

#6가지 column들 시각화 
# column_name_numerical=['vintage','age','dependents','city','customer_nw_category','branch_code'] 

# for i in range(len(column_name_numerical)):
#     plt.title(column_name_numerical[i])
#     plt.xlabel(column_name_numerical[i])    
#     plt.ylabel('Count')
#     if(i==2):
#         list=[0,1,2,3,4,5]
#         plt.hist(df['dependents'],list)
#         plt.show()
#         continue;
#     if(i==4):
#         list=[0,1,2,3,4,5]
#         plt.hist(df['customer_nw_category'],list)
#         plt.show()
#         continue;
#     numerical_vis_func(df[[column_name_numerical[i]]])


#3가지 categorical data 시각화 
# column_name_category=['gender','occupation','churn']
# for i in range(len(column_name_category)):
#     plt.title(column_name_category[i])
#     plt.xlabel(column_name_category[i])    
#     plt.ylabel('Count')
#     sns.countplot(df[column_name_category[i]])
#     plt.show() 

#Current value cloumns들 시각화 
#Current value visualization-------------------------------
column_name_current=['current_balance','current_month_credit','current_month_debit','current_month_balance']    
var_color_dict = {'current_balance': 'blue', 
                  'current_month_credit': 'red', 
                  'current_month_debit': 'yellow', 
                  'current_month_balance': 'green'}

i = [0, 0, 1, 1]
j = [0, 1, 0, 1]

f, axes = plt.subplots(2, 2, figsize=(8, 6),sharex=True)
for var, i, j in zip(var_color_dict, i, j):
    plt.xlabel(column_name_current[i])    
    sns.distplot(df[column_name_current[i]],color = var_color_dict[var],ax = axes[i, j])
plt.show()
#--------------------------------------------
#average value cloumns들 시각화 
# #average mothly balance value visualization-------------------------------
# column_name_average=['average_monthly_balance_prevQ','average_monthly_balance_prevQ2']    
# figure = plt.figure(figsize=(18,6))

# ax1 = plt.subplot(1,2,1)
# sns.distplot(df['average_monthly_balance_prevQ'], ax=ax1)
# ax2 = plt.subplot(1,2,2)
# sns.distplot(df['average_monthly_balance_prevQ2'], ax=ax2)    # 기본값은 kde(선) True, hist(막대) True
# plt.show()
# #--------------------------------------------

#previous value cloumns들 시각화 
# #previous value-------------------------------
# column_name_previous=['previous_month_end_balance','previous_month_credit','previous_month_debit','previous_month_balance']    
# var_color_dict2 = {'previous_month_end_balance': 'blue', 
#                   'previous_month_credit': 'red', 
#                   'previous_month_debit': 'yellow', 
#                   'previous_month_balance': 'green'}

# a = [0, 0, 1, 1]
# b = [0, 1, 0, 1]

# f, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
# for var2, a, b in zip(var_color_dict2, a, b):
#     sns.distplot(df[column_name_previous[a]],color = var_color_dict2[var2],ax = axes[a, b])

# plt.show()
# # ----------------------------------------


#월단위로 추출한 Last transaction 시각화 
# #----------Last transaction barplot with only month-----------------
# # Creating an instance(data) of Datetimeindex class using last_transaction
# temp_df=df.copy()
# temp_df['last_transaction'] = pd.DatetimeIndex(temp_df['last_transaction'])

# # There are only 3 values that last_transaction in 2018
# # All other records are in 2019 so drop 2018 values
# date = pd.DatetimeIndex(temp_df['last_transaction'])
# indexNames = temp_df[date.year == 2018].index
# temp_df.drop(indexNames, inplace=True)
# temp_df = temp_df.reset_index(drop=True)

# # Dealing with nan values in last_transaction
# date = pd.DatetimeIndex(temp_df['last_transaction'])
# temp_df['last_transaction'] = date.month
# temp_df.dropna(axis=0,inplace=True)
# print("last_transaction_month\n",temp_df['last_transaction'])

# plt.title("Last transation_month")
# plt.xlabel('Last transation_month')    
# plt.ylabel('Count')
# sns.countplot(temp_df['last_transaction'])
# plt.show() 
# #------------------------------------------------------


