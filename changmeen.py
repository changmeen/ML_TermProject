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

def findbest(best, max, min, jump): #베스트 찾고 위치 조정
    if best == 1: #best_n이 1일때, 1은 최소값이므로 점프값 조정하고 max값을 조정한다. 1~5 -> 1~3
        jump = (int)(jump / 2)
        max = best + jump + 1
    elif best + jump > max: #best_n이 pmax값에 가장 가까운 값일때, min값이 best_n값으로 시작해서 다시 조정(상승곡선을 그릴때)
        max = best + jump * 2 + 1
        min = best
    else: #중간에 최대값이 발견되서 조정할때, 1 4 7일때, 4가 최대값이면 4를 중심으로 3 4 5를 진행
        jump = (int)(jump / 2)
        max = best + jump + 1
        min = best - jump
    return max, min, jump

def knn(df, lbl):
    weights=['uniform', 'distance']
    w_best = 0
    w_best_w = 0
    w_best_p = 0
    w_best_n = 0
    for weight in weights:
        p_best = 0 #p best score
        p_best_p = 0 #p best p값
        p_best_n = 0 #p best n값
        for p in range(1, 3):
            n_min = 1 #neighbor 시작값
            n_max = 14 #neighbor 종료값
            g_n_min = n_min #neighbor 그래프 x눈금 시작값
            g_n_max = n_max #neighbor 그래프 x눈금 종료값
            n_jump = 4 #neighbor 점프값
            n_best = 0 #neighbor best score
            n_best_n = 0 #neighbor best n값
            n_base = 0 #neighbor 기준점 score
            n_base_n = 0 #neighbor 기준점 n값
            while True:
                n_all_score = []
                n_all_n = []
                for neighbor in range(n_min, n_max, n_jump):
                    if neighbor == n_base_n:  # n값이 base n값이면 이미 score값을 도출했으므로 continue
                        n_all_score.append(n_base)
                        n_all_n.append(n_base_n)
                        continue

                    if n_base_n == 0:  # 초기 base값 설정
                        n_base_n = neighbor

                    model = KNeighborsClassifier(n_neighbors=neighbor, p=p, weights=weight)
                    score = cross_val_score(model, df, lbl, cv=5)  # kfold #score값은 cv의 갯수만큼 존재하므로 평균값 필요시 mean()
                    print(
                        "Base n=" + str(n_base_n) + ", weight=" + str(weight) + ", p=" + str(p) + ", Current n=" + str(
                            neighbor) + ", Score=" + str(
                            score.mean()))

                    n_all_score.append(score.mean())
                    n_all_n.append(neighbor)

                    if n_base == 0:  # 시작값
                        n_base = score.mean()

                    if n_best < score.mean():
                        n_best = score.mean()  # 최대값
                        n_best_n = neighbor

                n_all_score = sorted(n_all_score, key=lambda all_score: n_all_n)
                n_all_n = sorted(n_all_n)
                plt.plot(n_all_n, n_all_score, marker='o')

                # if start * 1.0001 > best: #시작값보다 최대값이 K%보다 작을때 종료
                #    break;
                if n_jump == 1:  # 베스트값 구하고 jump값이 1일때 종료 -> 만약 3 4 5 진행했다면 jump값이 1이므로 종료
                    break;
                n_base = n_best  # 최소값 최대값이 바뀌었으니 베스트값을 시작값으로 하여 다시 비교
                n_base_n = n_best_n
                n_max, n_min, n_jump = findbest(n_best_n, n_max, n_min, n_jump)
                if n_max > g_n_max: #그래프 x눈금 종료값 확장
                    g_n_max = n_max
                if n_min < g_n_min: #그래프 x눈금 시작값 확장
                    g_n_min = n_min

            if p_best < n_best:
                p_best = n_best  # 최대값
                p_best_p = p
                p_best_n = n_best_n

            print("!weight=" + str(weight) + ", p=" + str(p) + ", Best n=" + str(n_best_n) + ", Best Score=" + str(n_best) + "!")  #p값일 때 최대값 출력

            plt.xticks(range(g_n_min, g_n_max))
            plt.show()

        if w_best < p_best:
            w_best = p_best  # 최대값
            w_best_w = weight
            w_best_p = p_best_p
            w_best_n = p_best_n

        print("@weight=" + str(weight) + ", Best p=" + str(p_best_p) + ", Best n=" + str(p_best_n) + ", Best Score=" + str(p_best) + "@")  #weight값일 때 최대값 출력
        
    print("#Best weight=" + str(w_best_w) + ", Best p=" + str(w_best_p) + ", Best n=" + str(w_best_n) + ", Best Score=" + str(
        w_best) + "#")  #knn 최대값 출력


def classifications(df):
    lbl=df['churn']
    df=df.drop(['churn'],axis=1)

    le = LabelEncoder()
    df['occupation'] = encoder(le, df['occupation'])

    st = StandardScaler()
    df = scaler(st, df)

    knn(df, lbl)

classifications(df)#호출호출