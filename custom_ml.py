from matplotlib import pyplot as plt
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
                 cv=None,
                 e=0):
        self.estimator=estimator #최상위 파라미터들 -> 언제든 self.로 불러올 수 있다.
        self.cv=cv
        self.param_grid=param_grid
        self.jump={}
        self.e_value=e+1
        self.e=False
        self.clus=False
        self.best_dict={}

    def dict_keys(self): #입력받은 파라미터들의 key값들 불러오기
        return list(self.param_grid.keys())

    def dict_val(self, repeat=1): #dict -> list하고 jump값이 존재하는 key값 따로 분류하고 None값 존재여부 확인
        args=list(self.param_grid.values()) #dict -> list
        for k in range(len(args)): #args의 key의 value값들을 차례대로 호출
            if(type(args[k][0])==list): # value값의 첫번째 값이 list값인지 확인 -> list값일 경우 jump값 존재한다고 판단
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
            else:
                jump = (int)(jump / 2)
                min = best - jump
                max = best + jump
        elif best == max:  #best값이 max일때 탐색 범위 확장
            if self.e: #오차가 앱실론미만일때 best가 max값이면 범위 좁히기(확장x)
                jump = (int)(jump / 2)
                max = best + jump
                min = best - jump
                self.e = False
            else:
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
        clus  = False
        if 'KMeans' in str(model):
            clus = True
        elif 'GaussianMixture' in str(model):
            clus = True
        elif 'MeanShift' in str(model):
            clus = True

        self.clus=clus

        # print(dict)
        if self.cv is not None:
            if y is not None:
                score = cross_val_score(model, X, y, cv=self.cv).mean()
            else:
                score = cross_val_score(model, X, cv=self.cv).mean()
        else:
            if y is not None:
                model.fit(X, y)
                if clus:
                    pred = model.predict(X)
                    score = silhouette_score(X, pred)
                else:
                    score = model.score(X, y)
            else:
                model.fit(X)
                if clus:
                    pred = model.predict(X)
                    score = silhouette_score(X,pred)
                else:
                    score = model.score(X)
        #print(score)

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

        visualize=[] #시각화
        vi_total = [] #범위 확장하거나 축소시 구분을 위함

        e=False
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
                        
                if score>=best: #최대값 갱신
                    best_i = args[k][i]

                    if best*self.e_value>score: #최대값 갱신할때 기존 최대값과 새로운 최대값의 오차가 앱실론 미만일때
                        if e: #이미 앱실론 미만인 경우가 존재했으면 2번째 이후는 더이상 계산하는게 의미가 없다고 판단
                            i = len(args[k]) - 1 #건너뛰기
                            e=False #초기화
                        else: #처음 발견된 경우 다음 것도 확인
                            e=True
                    else: #오차가 앱실론 미만일때 초기화
                        e=False
                    best = score
                    best_dict = dict

                q.append(score) #파라미터의 score값 저장
                t.append(q) #각 파라미터의 score값 저장

                visualize.append(q)

                if i==len(args[k])-1: #파라미터 score를 모두 구했을 때

                    vi_total.append(visualize) #구분을 위해 저장
                    visualize=[] #초기화

                    if best_i != None: #best 파라미터가 None이 아니며
                        if jump != None: #jump값이 존재할때
                            while jump>1: #범위를 좁혀가며 최적의 값 탐색
                                #self.e = e
                                min_i, max_i, jump = self.findbest(best_i, min_i, max_i, jump) #min max jump값 조정

                                more = list(range(min_i, max_i + 1, jump)) #추가된 파라미터값
                                if best_i in more:
                                    more.remove(best_i) #이미 계산된 파라미터 제거(best 파라미터)
                                more=filter(lambda a: a>0,more)

                                temp_best=[best_i,best] #best값을 제외하고 계산하였으므로 best값 추가
                                visualize.append(temp_best)

                                e = False
                                for w in more: #추가된 파라미터의 score 계산
                                    q = []
                                    dict[keys[k]] = w
                                    score = self.cal(dict, X, y)

                                    if score >= best:
                                        if best * self.e_value >= score:  # 최대값 갱신할때 기존 최대값과 새로운 최대값의 오차가 앱실론 미만일때
                                            if e:  # 이미 앱실론 미만인 경우가 존재했으면 2번째 이후는 더이상 계산하는게 의미가 없다고 판단
                                                e = False  # 초기화
                                                q.append(w)
                                                q.append(score)
                                                t.append(q)

                                                visualize.append(q)
                                                break
                                            else:  # 처음 발견된 경우 다음 것도 확인
                                                e = True
                                        else:  # 오차가 앱실론 미만일때 초기화
                                            e = False
                                        best = score
                                        best_i = w
                                        best_dict = dict.copy()
                                    q.append(w)
                                    q.append(score)
                                    t.append(q)

                                    visualize.append(q)

                                try:
                                    visualize = sorted(visualize, key=lambda visualize: visualize[0]) #섞인 파라미터 값들을 기준으로 score도 같이 sorted [13, 17, 15] -> [13, 15, 17]
                                except: #오류발생시
                                    for v in range(len(visualize)):
                                        if visualize[v][0] is None: #None을 0으로 치환
                                            visualize[v][0]=0
                                            visualize = sorted(visualize, key=lambda visualize: visualize[0])
                                            break

                                vi_total.append(visualize)
                                visualize = []

                    break
            else:
                result, score, dict = self.create(dict, k + 1, X, y) # 파라미터 구조도 생성 및 score 계산-> 재귀
                if best<=score: # 구조도 맨밑에 위치한 파라미터가 아닐때 해당 파라미터의 best값 갱신
                    best_i = args[k][i]

                    if best*self.e_value>score: # 최대값 갱신할때 기존 최대값과 새로운 최대값의 오차가 앱실론 미만일때
                        if e: # 이미 앱실론 미만인 경우가 존재했으면 2번째 이후는 더이상 계산하는게 의미가 없다고 판단
                            i = len(args[k]) - 1 #건너뛰기
                            e=False # 초기화
                        else: # 처음 발견된 경우 다음 것도 확인
                            e=True
                    else: # 오차가 앱실론 미만일때 초기화
                        e=False
                    best = score
                    best_dict = dict

                q.append(result)
                t.append(q)

                temp_w = [args[k][i], score]
                visualize.append(temp_w)

                if i==len(args[k])-1:

                    vi_total.append(visualize)
                    visualize = []

                    if best_i != None:
                        if jump != None:
                            while jump>1:
                                #self.e = e
                                min_i, max_i, jump = self.findbest(best_i, min_i, max_i, jump)
                                more = list(range(min_i, max_i + 1, jump))
                                if best_i in more:
                                    more.remove(best_i)  # 이미 계산된 파라미터 제거(best 파라미터)
                                more = filter(lambda a: a > 0, more)

                                temp_best = [best_i, best]
                                visualize.append(temp_best)

                                e=False
                                for w in more:
                                    q = []
                                    dict[keys[k]] = w
                                    result, score, dict = self.create(dict, k + 1, X, y) #추가된 파라미터의 구조도 생성 및 score 계산
                                    if best <= score:
                                        if best * self.e_value >= score:  # 최대값 갱신할때 기존 최대값과 새로운 최대값의 오차가 앱실론 미만일때
                                            if e:  # 이미 앱실론 미만인 경우가 존재했으면 2번째 이후는 더이상 계산하는게 의미가 없다고 판단
                                                e = False  # 초기화
                                                q.append(w)
                                                q.append(result)
                                                t.append(q)

                                                temp_w = [w, score]
                                                visualize.append(temp_w)

                                                break;
                                            else:  # 처음 발견된 경우 다음 것도 확인
                                                e = True
                                        else:  # 오차가 앱실론 미만일때 초기화
                                            e = False
                                        best_i = w
                                        best = score
                                        best_dict = dict.copy()

                                    q.append(w)
                                    q.append(result)
                                    t.append(q)

                                    temp_w = [w, score]
                                    visualize.append(temp_w)

                                try:
                                    visualize = sorted(visualize, key=lambda visualize: visualize[0])  # 섞인 파라미터 값들을 기준으로 score도 같이 sorted [13, 17, 15] -> [13, 15, 17]
                                except:  # 오류발생시
                                    for v in range(len(visualize)):
                                        if visualize[v][0] is None:  # None을 0으로 치환
                                            visualize[v][0] = 0
                                            visualize = sorted(visualize, key=lambda visualize: visualize[0])
                                            break


                                vi_total.append(visualize)
                                visualize = []
                    break
        self.e=False
        self.best_dict=best_dict

        """
        xticks=[] #눈금
        for v in range(len(vi_total)): #시각화를 위해 파라미터값과 score값 분리
            vi_x=[] #파라미터 값
            vi_y=[] #score값
            for temp in range(len(vi_total[v])):
                vi_x.append(vi_total[v][temp][0])
                vi_y.append(vi_total[v][temp][1])
                xticks.append(vi_total[v][temp][0])
            plt.plot(vi_x, vi_y, marker='o',markersize=5)
        """
        title_dict=best_dict.copy()
        temp_k=k
        """
        if None in xticks:
            xticks.remove(None)

        xticks=sorted(list(set(xticks))) #눈금 중복 제거 및 정렬
        """
        while True: #best_dict를 복사하고 상위 파라미터까지만 title로 함.
            try:
                del(title_dict[keys[temp_k]])
            except:
                break
            temp_k=temp_k+1
        """

        plt.title(str(self.estimator)+" "+str(title_dict))
        plt.xlabel(keys[k])
        plt.ylabel('Score')
        plt.xticks(xticks)
        plt.scatter(best_i, best, marker='*', s=100,zorder=100000,color='r')  # best값 따로 표기
        plt.show()
        if k!=0:
            print("Base Parameter : {}".format(title_dict))
        print("Learning Parameter : {}".format(keys[k]))
        print("Parameter Best Score : {}, {} : {}\n".format(best,keys[k],best_i))
        """

        return t, best, best_dict # 하위 파라미터의 생성된 파라미터 구조도, best값과 best 파라미터값들을 상위 파라미터에 전송

    def fit(self, X, y = None):
        dict={}
        return self.create(dict,0,X,y)

    def predict(self,X,y=None):
        base_estimator = clone(self.estimator)
        model = clone(base_estimator)
        model.set_params(**self.best_dict)
        if self.clus:
            model.fit(X)
        else:
            if y is not None:
                model.fit(X, y)
            else:
                model.fit(X)

        return model.predict(X), model

