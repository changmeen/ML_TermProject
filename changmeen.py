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
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from custom_ml import AutoML
from sklearn.decomposition import PCA

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


# def numerical_vis_func(x):
#     plt.hist(x,bins=10)
#     plt.show()

# ##6가지 column들 시각화 
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


# ##3가지 categorical data 시각화 
# column_name_category=['gender','occupation','churn']
# for i in range(len(column_name_category)):
#     plt.title(column_name_category[i])
#     plt.xlabel(column_name_category[i])    
#     plt.ylabel('Count')
#     sns.countplot(df[column_name_category[i]])
#     plt.show() 


# #Current value cloumns들 시각화 
# #Current value visualization-------------------------------
# column_name_current=['current_balance','current_month_credit','current_month_debit','current_month_balance']    
# var_color_dict = {'current_balance': 'blue', 
#                   'current_month_credit': 'red', 
#                   'current_month_debit': 'yellow', 
#                   'current_month_balance': 'green'}

# i = [0, 0, 1, 1]
# j = [0, 1, 0, 1]

# f, axes = plt.subplots(2, 2, figsize=(8, 6),sharex=True)
# for var, i, j in zip(var_color_dict, i, j):
#     plt.xlabel(column_name_current[i])    
#     sns.distplot(df[column_name_current[i]],color = var_color_dict[var],ax = axes[i, j])
# plt.show()
# #--------------------------------------------


# ##average value cloumns들 시각화 
# #average mothly balance value visualization-------------------------------
# column_name_average=['average_monthly_balance_prevQ','average_monthly_balance_prevQ2']    
# figure = plt.figure(figsize=(18,6))

# ax1 = plt.subplot(1,2,1)
# sns.distplot(df['average_monthly_balance_prevQ'], ax=ax1)
# ax2 = plt.subplot(1,2,2)
# sns.distplot(df['average_monthly_balance_prevQ2'], ax=ax2)    # 기본값은 kde(선) True, hist(막대) True
# plt.show()
# #--------------------------------------------

# ##previous value cloumns들 시각화 
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


# ##월단위로 추출한 Last transaction 시각화 
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
# #------------------------------------------------------


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

# print(str(df.isnull().sum()) + "\n")

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
"""
for i in outlier_list:
    print(i)
    #outlier_show(i)
"""
# Find outlier indexes
outlier_index = outlier(df, outlier_list)

# Deletion outlier values.
df = df.drop(outlier_index, axis=0).reset_index(drop=True)
df = df.drop(df[df['average_monthly_balance_prevQ2']<0].index, axis=0).reset_index(drop=True)
"""
for i in outlier_list:
    print(i)
    #outlier_show(i)
"""
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


def knn(df, lbl=None, e=0): # weight, p, neighbor
    n_neighbors = list(range(1, 20, 4))
    param = {
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'n_neighbors': [n_neighbors, 4]
    }

    t = KNeighborsClassifier()

    temp = AutoML(t, param_grid=param, cv=5,e=e)
    result,score,dict=temp.fit(df, lbl)

    #최적의 파라미터가 저장된 dict를 이용하여 모델 생성후 pred

    pred, model=temp.predict(df,lbl)

    print("---------{}---------".format(temp.estimator))
    print("Best Parameter : {}".format(dict))
    print("Best Score : {}\n".format(score))

    return pred, model


def dt(df, lbl=None, e=0): # criterion, max_depth, splitter
    max_depth=list(range(1,20,4))
    max_depth.insert(0, None)
    param = {
        'criterion':['gini', 'entropy'],
        'splitter':['best','random'],
        'max_depth': [max_depth, 4]
    }

    t=DecisionTreeClassifier(random_state=42)

    temp=AutoML(t,param_grid=param,cv=5,e=e)
    result,score,dict=temp.fit(df,lbl)

    pred, model = temp.predict(df, lbl)

    print("---------{}---------".format(temp.estimator))
    print("Best Parameter : {}".format(dict))
    print("Best Score : {}\n".format(score))

    return pred, model


def lr(df, lbl=None, e=0): # solver, penalty, C
    param = {
        'solver':['newton-cg','lbfgs','sag','saga'],
        'penalty':['none','l2'],
        'C':[100,10,1,0.1,0.01]
    }

    t=LogisticRegression(random_state=42)

    temp=AutoML(t,param_grid=param,cv=5,e=e)
    result,score,dict=temp.fit(df,lbl)

    pred, model = temp.predict(df, lbl)

    print("---------{}---------".format(temp.estimator))
    print("Best Parameter : {}".format(dict))
    print("Best Score : {}\n".format(score))

    return pred, model


def kmeans(df, lbl=None, e=0): # n_clusters, init, n_init, algorithm, max_iter
    param = {
        'n_clusters':[2],
        'init':['k-means++','random'],
        'algorithm':['full','elkan'],
    }

    t = KMeans(random_state=42)

    temp = AutoML(t, param_grid=param,e=e)
    result, score, dict = temp.fit(df,lbl)
    pred, model = temp.predict(df, lbl)

    print("---------{}---------".format(temp.estimator))
    print("Best Parameter : {}".format(dict))
    print("Best Score : {}\n".format(score))

    return pred, model


def gm(df, lbl=None,e=0): # n_components, covatiance_type, n_init, init_param, max_iter
    param = {
        'n_components':[2],
        'covariance_type':['full','tied','diag','spherical'],
        'init_params':['kmeans','random']
    }

    t = GaussianMixture(random_state=42)

    temp = AutoML(t, param_grid=param,e=e)
    result, score, dict = temp.fit(df,lbl)
    pred, model = temp.predict(df, lbl)

    print("---------{}---------".format(temp.estimator))
    print("Best Parameter : {}".format(dict))
    print("Best Score : {}\n".format(score))

    return pred, model


def meanshift(df, lbl=None,e=0): # n_components, covatiance_type, n_init, init_param, max_iter
    bandwidth = estimate_bandwidth(df)
    param = {
        'bandwidth': [bandwidth],
    }

    t = MeanShift()

    temp = AutoML(t, param_grid=param,e=e)
    result, score, dict = temp.fit(df,lbl)

    pred, model = temp.predict(df, lbl)

    print("---------{}---------".format(t))
    print("Best Parameter : {}".format(dict))
    print("Best Score : {}\n".format(score))

    return pred, model


def classifications(df):
    lbl=df['churn']
    df=df.drop(['churn'],axis=1)

    le = LabelEncoder()
    df['occupation'] = encoder(le, df['occupation'])

    st = StandardScaler()
    df = scaler(st, df)

    e=0.01

    pred1, model1= knn(df, lbl, e)
    pred2, model2 = dt(df, lbl, e)
    pred3, model3 = lr(df, lbl)

    list_pred = [pred1, pred2, pred3]
    model_names = ['KNN', 'DecisionTree', 'Logistic Regression']

    # 각 모델마다 confusion matrix와 classification report 생성
    for i, pred in enumerate(list_pred):
        print("The confusion matrix and classification report of", model_names[i])
        print(pd.DataFrame(confusion_matrix(lbl, pred)))
        print(classification_report(lbl, pred, target_names= ['Churn', 'Not Churn']))
        print('\n')

    model_list = [model1, model2, model3]
    Color = ['red','blue','green']
    plt.title("Roc Curve", fontsize =10)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # 3개 모델 ROC CURVE 그래프 생성
    for i, model in enumerate(model_list):
        prob = model.predict_proba(df)
        prob_positive = prob[:, 1]
        fpr, tpr, threshold = roc_curve(lbl, prob_positive)
        plt.plot(fpr, tpr, color = Color[i])
        plt.gca().legend(model_names, loc='lower right', frameon=True)

    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.show()

    return list_pred


def clustering(df):
    lbl = df['churn']
    #data = df.drop(['churn'], axis=1).copy()
    data=df.copy()

    le = LabelEncoder()
    data['occupation'] = encoder(le, data['occupation'])

    st = StandardScaler()
    data = scaler(st, data)
    e = 0.01

    pca = PCA(n_components=2)
    """
    pred1, model1 = kmeans(data, e=e)
    pc = pca.fit_transform(data)
    plt.title('KMeans')
    plt.scatter(pc[:, 0], pc[:, 1],c=pred1,s=10)
    plt.show()

    pred2, model2 = gm(data, e=e)
    pc = pca.fit_transform(data)
    plt.title('GaussianMixture')
    plt.scatter(pc[:, 0], pc[:, 1],c=pred2,s=10)
    plt.show()
    """
    pred3, model3 = meanshift(data, e=e)
    pc = pca.fit_transform(data)
    plt.title('MeanShift')
    plt.scatter(pc[:, 0], pc[:, 1],c=pred3,s=10)
    plt.show()

    list_pred = [pred1, pred2, pred3]
    return list_pred


# classification_results = classifications(df)
clustering_results = clustering(df)

# print("classification_results", classification_results)
print("clustering_results", clustering_results)