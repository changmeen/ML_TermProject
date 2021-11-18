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
    def __init__(self,              
                 estimator,
                 param_grid,
                 cv=None,
                 e=0):    #Get AutoML Parameters
        self.estimator=estimator #top-levels paramenters -> call self. at any time.
        self.cv=cv
        self.param_grid=param_grid
        self.jump={}
        self.e_value=e+1
        self.e=False
        self.clus=False
        self.best_dict={}

    def dict_keys(self): #Calling the key values ​​of the input parameters
        return list(self.param_grid.keys())

    def dict_val(self): #dict -> list, separate key values ​​where jump values ​​exist, and check whether None values ​​exist
        args=list(self.param_grid.values()) #dict -> list
        for k in range(len(args)): #Calls the values ​​of the key of args in sequence
            if(type(args[k][0])==list): # Check whether the first value of value is a list value -> If it is a list value, it is judged that a jump value exists
                args[k].extend(args[k][0]) #seperate list [[1,2,3],4]->[[1,2,3],4,1,2,3]

                self.jump[self.dict_keys()[k]]=args[k][1] #The second value is a jump value, so it is separated
                del args[k][0:2] #eliminate list, jump values [1,2,3]

                if None in args[k]: #Check for None value
                    args[k].remove(None) #delete if there is
                    args[k].sort() #sort
                    args[k].insert(0,None) #Put None-Value at the beginning

                else: #If there is nothing--> sort
                    args[k].sort()
        return args

    def dict_list(self): #change dict_var to list & return
        return list(self.dict_val())

    def findbest(self, best, min, max, jump):  # find best & location adjustment
        if best == min: #if best is min value
            max = best - jump
            if best == 1: #if best=1, search by 1~jump/2
                jump = (int)(jump / 2)
                max = best + jump
            else:
                jump = (int)(jump / 2)
                min = best - jump
                max = best + jump
        elif best == max:  #When the best value is max, the search range is extended.
            if self.e: #When the margin of error is less than absilon, if best is max, narrow the range (extension x)
                jump = (int)(jump / 2)
                max = best + jump
                min = best - jump
                self.e = False
            else:
                max = best + jump * 2
                min = best + jump
        else:  # When best value is less than max and greater than min
            jump = (int)(jump / 2)
            max = best + jump
            min = best - jump

        return min, max, jump

    def cal(self,dict,X,y): # Calculate AutoML score 
        base_estimator = clone(self.estimator)
        model = clone(base_estimator)
        model.set_params(**dict)
        clus  = False
        if 'KMeans' in str(model):
            clus = True
        elif 'GaussianMixture' in str(model):
            clus = True
        elif 'SpectralClustering' in str(model):
            clus = True

        self.clus=clus

        if self.cv is not None:
            if y is not None:
                score = cross_val_score(model, X, y, cv=self.cv).mean()
            else:
                score = cross_val_score(model, X, cv=self.cv).mean()
        else:
            if y is not None:
                if clus:
                    pred = model.fit_predict(X)
                    score = silhouette_score(X, pred)
                else:
                    model.fit(X, y)
                    score = model.score(X, y)
            else:
                if clus:
                    pred = model.fit_predict(X)
                    score = silhouette_score(X,pred)
                else:
                    model.fit(X)
                    score = model.score(X)

        return score

    def create(self,dict,k,X,y): #Create a parameter structure diagram -> recursion
        args = self.dict_list()
        keys = self.dict_keys()
        try: #If there is no jump value, the value is None.
            jump = self.jump.get(keys[k])
        except:
            jump = None

        score = 0 #Score comparison when the parameter at the bottom of the structure diagram has a jump value
        best_dict={} #The combination of parameters with the highest score
        best = 0 #best score
        best_i = 0 #Parameter value, when best score
        t=[] #Parameter structure diagram
        min_i=-1 #parameter starting value
        max_i=-1 #parameter end value

        visualize = []  # 시각화
        vi_total = []  # 범위 확장하거나 축소시 구분을 위함

        e=False
        for i in range(len(args[k])):

            q=[] #temp structure
            q.append(args[k][i]) #store parameter
            dict[keys[k]]=args[k][i] #create parameter dict

            if k == len(args) - 1: #When the structure diagram is the bottom parameter
                score = self.cal(dict, X,y) #Get parameter score

                if args[k][i]!=None: #Set min max value when not None
                    if min_i==-1:
                        min_i=args[k][i]
                    if max_i==-1:
                        max_i=args[k][i]
                    if min_i>args[k][i]:
                        min_i=args[k][i]
                    if max_i<args[k][i]:
                        max_i=args[k][i]

                if score>=best: #max update
                    best_i = args[k][i]
                    if jump != None:
                        if best * self.e_value >= score:  # When the maximum value is updated, when the error between the old maximum value and the new maximum value is less than absilon
                            if e:  # If there has already been a case of less than Absilon, it is judged that it is meaningless to calculate any more after the second
                                i = len(args[k]) - 1  # Skip
                                e = False  # reset
                            else:  # If found for the first time, also check
                                e = True
                        else:  # Reset when the error is less than Absilon
                            e = False
                    best = score
                    best_dict = dict.copy()

                q.append(score) #Save the score value of the parameter
                t.append(q) #Save the score of each parameter

                visualize.append(q)

                if i==len(args[k])-1: #When all parameter scores are obtained

                    vi_total.append(visualize)  # 구분을 위해 저장
                    visualize = []  # 초기화

                    if best_i != None: #best parameter is not None
                        if jump != None: #When a jump value exists
                            while jump>1: #Narrow the range to find the optimal value
                                min_i, max_i, jump = self.findbest(best_i, min_i, max_i, jump) #Adjust min max jump value

                                more = list(range(min_i, max_i + 1, jump)) #Added parameter value
                                if best_i in more:
                                    more.remove(best_i) #Remove already calculated parameters (best parameters)
                                more=filter(lambda a: a>0,more)

                                temp_best = [best_i, best]  # best값을 제외하고 계산하였으므로 best값 추가
                                visualize.append(temp_best)

                                e = False
                                for w in more: #Calculate the score of the added parameter
                                    q = []
                                    dict[keys[k]] = w
                                    score = self.cal(dict, X, y)

                                    if score >= best:
                                        if best * self.e_value >= score:  # When the maximum value is updated, when the error between the old maximum value and the new maximum value is less than absilon
                                            if e:  # If there has already been a case of less than Absilon, it is judged that it is meaningless to calculate any more after the second
                                                e = False  # reset
                                                q.append(w)
                                                q.append(score)
                                                t.append(q)

                                                visualize.append(q)
                                                break
                                            else:  # If found for the first time, also check
                                                e = True
                                        else:  # Reset when the error is less than Absilon
                                            e = False
                                        best = score
                                        best_i = w
                                        best_dict = dict.copy()
                                    q.append(w)
                                    q.append(score)
                                    t.append(q)

                                    visualize.append(q)

                                try:
                                    visualize = sorted(visualize, key=lambda visualize: visualize[
                                        0])
                                except:
                                    for v in range(len(visualize)):
                                        if visualize[v][0] is None:
                                            visualize[v][0] = 0
                                            visualize = sorted(visualize, key=lambda visualize: visualize[0])
                                            break

                                vi_total.append(visualize)
                                visualize = []


                    break
            else:
                result, score, dict = self.create(dict.copy(), k + 1, X, y) # Create parametric diagram and calculate score -> recursion
                if best<=score: # When it is not a parameter located at the bottom of the structure diagram, the best value of the corresponding parameter is updated
                    best_i = args[k][i]
                    if jump!=None:
                        if best * self.e_value >= score:  # When the maximum value is updated, when the error between the old maximum value and the new maximum value is less than absilon
                            if e:  # If there has already been a case of less than Absilon, it is judged that it is meaningless to calculate any more after the second
                                i = len(args[k]) - 1  # skip
                                e = False  # reset
                            else:  # If found for the first time, also check
                                e = True
                        else:  # Reset when the error is less than Absilon
                            e = False
                    best = score
                    best_dict = dict.copy()

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
                                min_i, max_i, jump = self.findbest(best_i, min_i, max_i, jump)
                                more = list(range(min_i, max_i + 1, jump))
                                if best_i in more:
                                    more.remove(best_i)  # Remove already calculated parameters (best parameters)
                                more = filter(lambda a: a > 0, more)

                                temp_best = [best_i, best]
                                visualize.append(temp_best)

                                e=False
                                for w in more:
                                    q = []
                                    dict[keys[k]] = w
                                    result, score, dict = self.create(dict.copy(), k + 1, X, y) #Create a structure diagram of the added parameter and calculate the score
                                    if best <= score:
                                        if best * self.e_value >= score:  # When the maximum value is updated, when the error between the old maximum value and the new maximum value is less than absilon
                                            if e:  # If there has already been a case of less than Absilon, it is judged that it is meaningless to calculate any more after the second
                                                e = False  # reset
                                                q.append(w)
                                                q.append(result)
                                                t.append(q)

                                                temp_w = [w, score]
                                                visualize.append(temp_w)
                                                break;
                                            else:  # If found for the first time, also check
                                                e = True
                                        else:  # Reset when the error is less than Absilon
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
                                    visualize = sorted(visualize, key=lambda visualize: visualize[0])
                                except:  # 오류발생시
                                    for v in range(len(visualize)):
                                        if visualize[v][0] is None:
                                            visualize[v][0] = 0
                                            visualize = sorted(visualize, key=lambda visualize: visualize[0])
                                            break

                                vi_total.append(visualize)
                                visualize = []

                    break
        self.e=False

        self.best_dict=best_dict.copy()
        xticks = []  # 눈금
        for v in range(len(vi_total)):
            vi_x = []
            vi_y = []
            for temp in range(len(vi_total[v])):
                if jump!=None:
                    if vi_total[v][temp][0]==None:
                        vi_total[v][temp][0]=0
                vi_x.append(vi_total[v][temp][0])
                vi_y.append(vi_total[v][temp][1])
                xticks.append(vi_total[v][temp][0])
            plt.plot(vi_x, vi_y, marker='o', markersize=5)
        title_dict = best_dict.copy()
        temp_k = k

        xticks = sorted(list(set(xticks)))

        while True:
            try:
                del (title_dict[keys[temp_k]])
            except:
                break
            temp_k = temp_k + 1

        plt.title(str(self.estimator) + " " + str(title_dict))
        plt.xlabel(keys[k])
        plt.ylabel('Score')
        plt.xticks(xticks)
        plt.scatter(best_i, best, marker='*', s=100, zorder=100000, color='r')  # best값 따로 표기
        plt.show()

        if k != 0:
            print("Base Parameter : {}".format(title_dict))
        print("Learning Parameter : {}".format(keys[k]))
        print("Parameter Best Score : {}, {} : {}\n".format(best, keys[k], best_i))

        return t, best, best_dict.copy() # Structure diagram of created parameter of sub-parameter, Send the best value and best parameter values ​​to the upper parameter

    def fit(self, X, y = None):
        dict={}
        return self.create(dict,0,X,y)

    def predict(self,X,y=None):
        base_estimator = clone(self.estimator)
        model = clone(base_estimator)
        model.set_params(**self.best_dict)
        if self.clus:
            return model.fit_predict(X), model
        else:
            if y is not None:
                model.fit(X, y)
            else:
                model.fit(X)

        return model.predict(X), model

