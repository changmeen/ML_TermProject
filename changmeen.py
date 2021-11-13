import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn.metrics as metrics
from collections import Counter
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, estimate_bandwidth, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from custom_ml import AutoML
from sklearn.decomposition import PCA

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings('ignore')
sns.set(style='white')


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


def encoder(encoder, df):
    return encoder.fit_transform(df)


def scaler(scaler, df):
    scaled_features = scaler.fit_transform(df)
    return pd.DataFrame(scaled_features, index=df.index, columns=df.columns)


def knn(df, lbl=None, e=0): # weight, p, neighbor
    n_neighbors = list(range(1, 20, 4))
    param = {
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'n_neighbors': [n_neighbors, 4]
    }

    t = KNeighborsClassifier()

    temp = AutoML(t, param_grid=param, cv=5,e=e)
    result, score, dict = temp.fit(df, lbl)

    # 최적의 파라미터가 저장된 dict를 이용하여 모델 생성후 pred

    pred, model = temp.predict(df,lbl)

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
        'n_clusters':[2],
        'eigen_solver':['arpack','lobpcg','amg',None]
    }

    t = SpectralClustering()

    temp = AutoML(t, param_grid=param,e=e)
    result, score, dict = temp.fit(df,lbl)

    pred, model = temp.predict(df, lbl)

    print("---------{}---------".format(t))
    print("Best Parameter : {}".format(dict))
    print("Best Score : {}\n".format(score))

    return pred, model


def classifications(df, lbl):
    e=0.01

    pred1, model1= knn(df, lbl, e)
    pred2, model2 = dt(df, lbl, e)
    pred3, model3 = lr(df, lbl)

    list_pred = [pred1, pred2, pred3]
    model_names = ['KNN', 'DecisionTree', 'Logistic Regression']

    # 각 모델마다 confusion matrix와 classification report 생성
    for i, pred in enumerate(list_pred):
        print("The confusion matrix and classification report of", model_names[i])
        print('accuracy', metrics.accuracy_score(lbl, pred))
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

# Find outlier indexes
outlier_index = outlier(df, outlier_list)

# Deletion outlier values.
df = df.drop(outlier_index, axis=0).reset_index(drop=True)
df = df.drop(df[df['average_monthly_balance_prevQ2']<0].index, axis=0).reset_index(drop=True)

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
df = df.drop(col, axis=1)

# Drop customer_id
df.drop(['customer_id'], axis=1, inplace=True)

# df는 원본 데이터 data는 원본에서 churn을 땐거
lbl = df['churn']
data = df.drop(['churn'], axis=1)

le = LabelEncoder()
data['dependents'] = encoder(le, data['dependents'])
data['city'] = encoder(le, data['city'])
data['gender'] = encoder(le, data['gender'])
data['occupation'] = encoder(le, data['occupation'])

st = StandardScaler()
data = scaler(st, data)
print(data)

# Feature Selection(Using Select-KBest)-----------------
X = data
y = lbl

selector = SelectKBest(score_func=f_classif, k=17)
fit = selector.fit(X, y)

dfcolumns = pd.DataFrame(X.columns)
dfscores = pd.DataFrame(fit.scores_)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Spec', 'Score']

# print(featureScores.nlargest(17, 'Score'))
# ---------------------------------------------------------

selected_features = ['current income to spending ratio',
                     'current total income', 'current total spending',
                     'previous income to spending ratio',
                     'current credit usage', 'previous total spending',
                     'previous total income', 'previous credit usage'
                     ]
data = data[selected_features]

classification_results = classifications(data, lbl)
# clustering_results = clustering(df)

print("classification_results", classification_results)
# print("clustering_results", clustering_results)