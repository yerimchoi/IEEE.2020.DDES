#########################################################################################################################
### Project  :  Distribution-based Dynamic Ensemble Selection
### Script   :  DDES on GIT.py
### Contents :  Creating Region of Competence considering data distribution
#########################################################################################################################

#########################################################################################################################
# Setting up Environment
#########################################################################################################################
import os
import numpy as np
import pandas as pd
import random 
import warnings
from collections import Counter

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#########################################################################################################################
# DDES functions
#########################################################################################################################
def randomlist(list):
    index = len(list) - 1
    value = list[random.randint(0, index)]
    return value

def define_model(model_name):
    global model
    i = random.randint(1, 42)
    
    # neural network
    activation_list = ['relu', 'tanh', 'logistic']
    hidden_list = [(10,), (10, 10), (10, 10, 10), (100,), (100, 100)]
    nn_solver_list = ['lbfgs', 'sgd', 'adam']
    
    # svm
    kernel_list = ['linear', 'rbf']

    # decision tree
    max_feature_list = ['auto', 'sqrt', 'log2']
    criterion_list = ['entropy', 'gini']
    
    # logistic regression
    lr_solver_list = ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga']
    
    # knn
    neigh_list = [3, 5, 7, 9]
    weigh_list = ['uniform', 'distance']
    
    # XGBoost
    booster_list = ['gbtree', 'gblinear', 'dart']

    if model_name == 'nn':
        model = MLPClassifier(activation = randomlist(activation_list), 
                              hidden_layer_sizes = randomlist(hidden_list),
                              solver = randomlist(nn_solver_list), 
                              random_state = i)
        
    elif model_name == 'svc':
        model = SVC(C = round(random.uniform(1, 15),2), 
                    gamma = round(random.uniform(0.01, 1), 2),
                    kernel = randomlist(kernel_list),
                    random_state = i)
        
    elif model_name == 'dt':
        model = DecisionTreeClassifier(max_features = randomlist(max_feature_list), 
                                       criterion = randomlist(criterion_list),
                                       min_samples_split = 2, 
                                       min_samples_leaf = 1,
                                       random_state = i)
        
    elif model_name == 'lr':
        model = LogisticRegression(solver = randomlist(lr_solver_list),
                                   random_state = i)
        
    elif model_name == 'knn':
        model = KNeighborsClassifier(n_neighbors = randomlist(neigh_list),
                                     weights = randomlist(weigh_list))
        
    elif model_name == 'gb':
        model = GradientBoostingClassifier(learning_rate = 0.1,
                                           min_samples_split = 2,
                                           min_samples_leaf = 1,
                                           random_state = i)
        
    elif model_name == 'xgb':
        model = XGBClassifier(eta = round(random.uniform(0, 1), 2),
                              gamma = round(random.uniform(0.01, 1), 2),
                              booster = randomlist(booster_list),
                              random_state = i)
    
    elif model_name == 'nb':
        model = GaussianNB()

    return model


class DDES:
    def __init__(self, file, kmin, accthreshold):
        self.file = file
        self.kmin = kmin
        self.accthreshold = accthreshold
        
    def _load_data(self):
        x = self.file.drop(['class'], axis=1)
        y = self.file['class']
        
        try:
            self.train_xo, self.test_x, self.train_yo, self.test_y = train_test_split(x, y, test_size = 0.25, stratify = y)
            self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.train_xo, self.train_yo, test_size = 0.3, stratify = self.train_yo)
            
            ss = StandardScaler()
            self.train_xo = ss.fit_transform(self.train_xo)
            self.train_x, self.val_x, self.test_x = ss.transform(self.train_x), ss.transform(self.val_x), ss.transform(self.test_x)
            self.train_y, self.val_y, self.test_y = np.array(self.train_y), np.array(self.val_y), np.array(self.test_y)

            return 0
            
        except ValueError:
            return 1
            
    def _make_base_pool(self):
        models = ['nn']*11 + ['svc']*12 + ['dt']*6 + ['lr']*5 + ['gb']*5 +\
                 ['knn']*6 + ['xgb']*6 + ['nb']
        
        self.base_pool_list = [define_model(x) for x in models]
        
        self.base_pool = []
        for base in self.base_pool_list:
            self.base_pool.append(base.fit(self.train_x, self.train_y))
               
    def ddesI(self):
        self._make_base_pool()
        
        test_pred = []
        ite = 0
        
        for x in self.test_x:
            self.std = np.std(self.train_x, axis = 0)
            
            if 0 in self.std:
                indices = list(np.where(self.std == 0)[0])
                for ind in indices:
                    self.std[ind] = 1
                    
            breakindex = 0
            base_pool = []
            
            while True:         
                s = 1
                std = self.std
                
                while True:                        
                    r_valx, r_valy = [], []
                    base_pool = []
                    
                    if s >= 3:
                        breakindex = 30
                        break
                    
                    for i, val in enumerate(self.val_x):
                        ellip = np.sum(((val-x)/std)**2)
                                           
                        if ellip <= 1:
                            r_valx.append(val)
                            r_valy.append(self.val_y[i])
                            
                    if len(r_valx) < self.kmin:
                        s += 0.1
                        std = self.std * s
                        continue
                
                    else:
                        break

                if breakindex == 30:
                    base_pool = self.base_pool
                    break
                
                for base in self.base_pool:
                    pred_list = []

                    for p in r_valx:
                        pred = base.predict(p.reshape(1, -1))[0]
                        pred_list.append(pred)
                        
                    correct = np.where(np.array(pred_list) == r_valy)[0]
                    acc = len(correct) / len(r_valy)
                    
                    if acc >= self.accthreshold:
                        base_pool.append(base)
                
                if (len(base_pool)) > 0:
                    break
                
                else:
                    self.kmin += 1
                    breakindex += 1                    
                    continue

            final_pred = []
            for c in base_pool:
                pred = c.predict(x.reshape(1, -1))[0]
                final_pred.append(pred)
            
            fin = Counter(final_pred)

            if len(fin.keys()) == 1:
                test_pred.append(list(fin.keys())[0])
                
            else:
                maj_class = list(fin.keys())[0]
                maj_num = fin[maj_class]
                same_num = [maj_class]
                
                for cl in list(fin.keys())[1:]:
                    if fin[cl] > maj_num:
                        maj_class = cl
                        maj_num = fin[cl]
                    elif fin[cl] < maj_num:
                        continue
                    elif fin[cl] == maj_num:
                        same_num.append(cl)

                if len(same_num) == 1:
                    test_pred.append(maj_class)
                    
                elif len(same_num) > 1:
                    random_class = random.choice(same_num)
                    test_pred.append(random_class)
            
            ite += 1

        final_acc = accuracy_score(np.array(test_pred), self.test_y)
        return final_acc 
    
    def ddesM(self):
        self._make_base_pool()
        
        test_pred = []
        ite = 0
        
        for x in self.test_x:
            breakindex = 0
            base_pool = []
            
            while True:
                if breakindex == 10:
                    base_pool = self.base_pool
                    break
                               
                nn = NearestNeighbors(algorithm='brute', 
                                metric='mahalanobis', 
                                metric_params={'V': np.cov(self.val_x)})
                nn.fit(self.val_x)
                mah_list = nn.kneighbors(np.array(x).reshape(1,-1), self.kmin, return_distance=False)
                
                r_valx = self.val_x[mah_list, :][0]
                r_valy = self.val_y[mah_list][0]

                for base in self.base_pool:
                    pred_list = []

                    for p in r_valx:
                        pred = base.predict(p.reshape(1, -1))[0]
                        pred_list.append(pred)
                        
                    correct = np.where(np.array(pred_list) == r_valy)[0]
                    acc = len(correct) / len(r_valy)
                    
                    if acc >= self.accthreshold:
                        base_pool.append(base)
                
                if (len(base_pool)) > 0:
                    break
                
                else:
                    self.kmin += 1
                    breakindex += 1                    
                    continue

            final_pred = []
            for c in base_pool:
                pred = c.predict(x.reshape(1, -1))[0]
                final_pred.append(pred)
            
            fin = Counter(final_pred)

            if len(fin.keys()) == 1:
                test_pred.append(list(fin.keys())[0])
                
            else:
                maj_class = list(fin.keys())[0]
                maj_num = fin[maj_class]
                same_num = [maj_class]
                
                for cl in list(fin.keys())[1:]:
                    if fin[cl] > maj_num:
                        maj_class = cl
                        maj_num = fin[cl]
                    elif fin[cl] < maj_num:
                        continue
                    elif fin[cl] == maj_num:
                        same_num.append(cl)

                if len(same_num) == 1:
                    test_pred.append(maj_class)
                    
                elif len(same_num) > 1:
                    random_class = random.choice(same_num)
                    test_pred.append(random_class)
            
            ite += 1

        final_acc = accuracy_score(np.array(test_pred), self.test_y)
        return final_acc 

#########################################################################################################################
# Experiment
#########################################################################################################################
if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')
    file_path = 'file_path'
    data_path = 'data_path'

    data_list = os.listdir(data_path)
    data_list.sort()

    for data in data_list:
        csv_file = pd.read_csv(data_path + data)

        result_ddesi = []
        result_ddesm = []
        
        for _ in range(100):
            while True:
                try:
                    print(data)
                    print(">>> {} th iteration".format(_))
          
                    ddes = DDES(csv_file, kmin = 7, accthreshold = 0.3)
                    data_index = ddes._load_data()
                  
                    if data_index == 1:
                        continue
                  
                    else:            
                        ddesi_acc = ddes.ddesI()
                        ddesm_acc = ddes.ddesM()
                        
                        result_ddesi.append(ddesi_acc)
                        result_ddesm.append(ddesm_acc)
                      
                        df_ddesi = pd.DataFrame(result_ddesi, columns = ['ddesi'])
                        df_ddesm = pd.DataFrame(result_ddesm, columns = ['ddesm'])
                        
                        df_ddesi.to_csv(file_path + 'ddesi_' + data, index = False, header = True)
                        df_ddesm.to_csv(file_path + 'ddesm_' + data, index = False, header = True)

                        break
          
                except (np.linalg.LinAlgError, ZeroDivisionError):
                    continue
