from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from sklearn.base import clone

class AutoML:
    def __init__(self, #AutoML 파라미터 받기
                 estimator,
                 param_grid,
                 cv=None):
        self.estimator=estimator #최상위 파라미터들 -> 언제든 self.로 불러올 수 있다.
        self.cv=cv
        self.param_grid=param_grid
        self.jump={}
        self.dict=[]

    def dict_keys(self): #입력받은 파라미터들의 key값들 불러오기
        return list(self.param_grid.keys())

    def dict_val(self, repeat=1): #dict -> list하고 jump값이 존재하는 key값 따로 분류하고 None값 존재여부 확인
        args=list(self.param_grid.values()) #dict -> list
        for k in range(len(args)): #args의 key의 value값들을 차례대로 호출
            if(type(args[k][0])==list): #value값의 첫번째 값이 list값인지 확인 -> list값일 경우 jump값 존재한다고 판단
                args[k].extend(args[k][0]) #list를 각 값으로 분해 [[1,2,3],4]->[[1,2,3],4,1,2,3]

                self.jump[self.dict_keys()[k]]=args[k][1] #두번째값은 jump값이므로 따로 분리
                del args[k][0:2] #list, jump값 제거하기 [1,2,3]

                if None in args[k]: #None값 있는지 확인하기
                    args[k].remove(None) #있으면 지우고
                    args[k].sort() #sort하고
                    args[k].insert(0,None) #맨앞에 None값 넣어주기

                else: #없으면 걍 sort
                    args[k].sort()
        return args

    def dict_list(self): #dict_val list화해서 return
        return list(self.dict_val())

    def findbest(self, best, min, max, jump):  # 베스트 찾고 위치 조정
        if best == min: #best가 최소값일떄
            max = best - jump
            if best == 1: #best가 1이면 1~jump/2만큼 탐색
                jump = (int)(jump / 2)
                max = best + jump
            elif best - jump * 2 > 0: #min값 조정(음수x)
                min = best - jump * 2
            elif best - jump > 0: #min값 조정(음수x)
                min = best - jump
            else: #min값 조정이 안될때 jump, max값 조정
                jump = (int)(jump / 2)
                min = best
                max = best + jump
        elif best == max:  #best값이 max일때 탐색 범위 확장
            max = best + jump * 2
            min = best + jump
        else:  # best가 max보다 작고 min값보다 클때
            jump = (int)(jump / 2)
            max = best + jump
            min = best - jump

        return min, max, jump

    def cal(self,dict,X,y): #AutoML score 계산
        base_estimator = clone(self.estimator)
        model = clone(base_estimator)
        model.set_params(**dict)
        clus = False
        if 'KMeans' in str(model):
            clus = True
        elif 'GaussianMixture' in str(model):
            clus = True
        elif 'MeanShift' in str(model):
            clus = True
        if self.cv is not None:
            if y is not None:
                score = cross_val_score(model, X, y, cv=self.cv).mean()
            else:
                score = cross_val_score(model, X, cv=self.cv).mean()
        else:
            if y is not None:
                print(model)
                model.fit(X, y)
                if clus:
                    score = silhouette_score(X, y)
                else:
                    score = model.score(X, y)
                print(score)
            else:
                print(model)
                model.fit(X)
                if clus:
                    pred = model.predict(X)
                    score = silhouette_score(X,pred)
                else:
                    score = model.score(X)
                print(score)

        return score

    def create(self,dict,k,X,y): #파라미터 구조도 생성 -> 재귀
        args = self.dict_list()
        keys = self.dict_keys()
        try: #jump값 없으면 None값
            jump = self.jump.get(keys[k])
        except:
            jump = None

        score = 0 #구조도 맨밑의 파라미터가 jump값 존재시 score비교
        best_dict={} #score가 가장 높은 파라미터 조합
        best = 0 #best score
        best_i = 0 #best score일때 파라미터 값
        t=[] #파라미터 구조도
        min_i=-1 #파라미터 시작값
        max_i=-1 #파라미터 종료값
        for i in range(len(args[k])):

            q=[] #temp 구조도
            q.append(args[k][i]) #파라미터 저장
            dict[keys[k]]=args[k][i] #파라미터 dict 생성

            if k == len(args) - 1: #구조도 맽밑 파라미터일때
                score = self.cal(dict, X,y) #파라미터 score 구하기

                if args[k][i]!=None: #None값이 아닐때 min max값 설정
                    if min_i==-1:
                        min_i=args[k][i]
                    if max_i==-1:
                        max_i=args[k][i]
                    if min_i>args[k][i]:
                        min_i=args[k][i]
                    if max_i<args[k][i]:
                        max_i=args[k][i]

                if score>best: #최대값 갱신
                    best=score
                    best_i=args[k][i]
                    best_dict=dict

                q.append(score) #파라미터의 score값 저장
                t.append(q) #각 파라미터의 score값 저장
                if i==len(args[k])-1: #파라미터 score를 모두 구했을 때
                    if best_i != None: #best 파라미터가 None이 아니며
                        if jump != None: #jump값이 존재할때
                            while jump>1: #범위를 좁혀가며 최적의 값 탐색
                                min_i, max_i, jump = self.findbest(best_i, min_i, max_i, jump) #min max jump값 조정
                                more = list(range(min_i, max_i + 1, jump)) #추가된 파라미터값
                                if best_i in more:
                                    more.remove(best_i) #이미 계산된 파라미터 제거(best 파라미터)
                                more=filter(lambda a: a>0,more)
                                for w in more: #추가된 파라미터의 score 계산
                                    q = []
                                    dict[keys[k]] = w
                                    score = self.cal(dict, X, y)
                                    if score > best:
                                        best = score
                                        best_i = w
                                        best_dict = dict.copy()

                                    q.append(w)
                                    q.append(score)
                                    t.append(q)

            else:
                result,score,dict = self.create(dict, k + 1, X, y) #파라미터 구조도 생성 및 score 계산-> 재귀
                if best<score: #구조도 맨밑에 위치한 파라미터가 아닐때 해당 파라미터의 best값 갱신
                    best_i = args[k][i]
                    best=score
                    best_dict=dict

                q.append(result)
                t.append(q)

                if i==len(args[k])-1:
                    if best_i != None:
                        if jump != None:
                            while jump>1:
                                min_i, max_i, jump = self.findbest(best_i, min_i, max_i, jump)
                                more = list(range(min_i, max_i + 1, jump))
                                if best_i in more:
                                    more.remove(best_i)  # 이미 계산된 파라미터 제거(best 파라미터)
                                more = filter(lambda a: a > 0, more)
                                for w in more:
                                    q = []
                                    dict[keys[k]] = w
                                    result, score, dict = self.create(dict, k + 1, X, y) #추가된 파라미터의 구조도 생성 및 score 계산
                                    if best < score:
                                        best_i = w
                                        best = score
                                        best_dict = dict.copy()

                                    q.append(w)
                                    q.append(result)
                                    t.append(q)


        return t,best,best_dict #하위 파라미터의 생성된 파라미터 구조도, best값과 best 파라미터값들을 상위 파라미터에 전송



    def fit(self, X, y = None):
        dict={}
        return self.create(dict,0,X,y)

    #def fit(self, X, y=None):