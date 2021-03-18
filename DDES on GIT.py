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

from deslib.dcs import OLA, MCB, LCA
from deslib.des import KNORAE, KNORAU, DESP, METADES

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
                    probability = True,
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

class DES:
    def __init__(self, file, k, accthreshold):
        self.file = file
        self.k = k
        self.accthreshold = accthreshold
        
    def _load_data(self):
        x = self.file.drop(['class'], axis=1)
        y = self.file['class']

        self.train_xo, self.test_x, self.train_yo, self.test_y = train_test_split(x, y, test_size = 0.25, stratify = y)
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.train_xo, self.train_yo, test_size = 0.3, stratify = self.train_yo)

        ss = StandardScaler()
        self.train_xo = ss.fit_transform(self.train_xo)
        self.train_x, self.val_x, self.test_x = ss.transform(self.train_x), ss.transform(self.val_x), ss.transform(self.test_x)
        self.train_y, self.val_y, self.test_y = np.array(self.train_y), np.array(self.val_y), np.array(self.test_y)
            
    def _make_base_pool(self):
        models = ['nn']*7 + ['svc']*7 + ['dt']*7 + ['lr']*7 + ['gb']*7 +\
                 ['knn']*7 + ['xgb']*7 + ['nb']*2
        
        self.base_pool_list = [define_model(x) for x in models]
        
        self.base_pool = []
        for base in self.base_pool_list:
            self.base_pool.append(base.fit(self.train_x, self.train_y))
               
    def ddesI(self):
        self.std = np.std(self.train_x, axis = 0)
        if 0 in self.std:
                indices = list(np.where(self.std == 0)[0])
                for ind in indices:
                    self.std[ind] = 1
                    
        test_pred = []
        ite = 0
        
        for x in self.test_x:    
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
                            
                    if len(r_valx) < self.k:
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
                    self.k += 1
                    breakindex += 1                    
                    continue

            final_pred = []
            for c in base_pool:
                pred = c.predict(x.reshape(1, -1))[0]
                final_pred.append(pred)
            
            test_pred.append(max(set(final_pred), key = final_pred.count))
            
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
                mah_list = nn.kneighbors(np.array(x).reshape(1,-1), self.k, return_distance=False)
                
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
                    self.k += 1
                    breakindex += 1                    
                    continue

            final_pred = []
            for c in base_pool:
                pred = c.predict(x.reshape(1, -1))[0]
                final_pred.append(pred)
                        
            test_pred.append(max(set(final_pred), key = final_pred.count))
            
            ite += 1

        final_acc = accuracy_score(np.array(test_pred), self.test_y)
        return final_acc 
    
    def other_des(self):
        # DES
        knorau = KNORAU(self.base_pool, IH_rate = 0.0)
        knorae = KNORAE(self.base_pool, IH_rate = 0.0)
        desp = DESP(self.base_pool, IH_rate = 0.0)
        metades = METADES(self.base_pool, IH_rate = 0.0)
        # DCS 
        lca = LCA(self.base_pool)
        ola = OLA(self.base_pool)
        mcb = MCB(self.base_pool)
        
        # Fitting DES and DCS        
        knorau.fit(self.val_x, self.val_y)
        knorae.fit(self.val_x, self.val_y)
        desp.fit(self.val_x, self.val_y)
        metades.fit(self.val_x, self.val_y)
        lca.fit(self.val_x, self.val_y)
        ola.fit(self.val_x, self.val_y)
        mcb.fit(self.val_x, self.val_y)
        
        acc_knorau = knorau.score(self.test_x, self.test_y)
        acc_knorae = knorae.score(self.test_x, self.test_y)
        acc_desp = desp.score(self.test_x, self.test_y)
        acc_metades = metades.score(self.test_x, self.test_y)
        acc_lca = lca.score(self.test_x, self.test_y)
        acc_ola = ola.score(self.test_x, self.test_y)
        acc_mcb = mcb.score(self.test_x, self.test_y)
        
        result = [acc_knorau, acc_knorae, acc_desp, acc_metades, acc_lca, acc_ola, acc_mcb]
        return result 

#########################################################################################################################
# Experiment
#########################################################################################################################
if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')
    file_path = 'file_path'
    data_path = 'data_path'

    data_list = os.listdir(data_path)
    data_list.sort()
    result = []
    
    for data in data_list:
        csv_file = pd.read_csv(data_path + data)
        each_result = []

        for _ in range(100):
            print(data)
            print(">>> {} th iteration".format(_))

            des = DES(csv_file, k = 7, accthreshold = 0.3)
            des._load_data()
            des._make_base_pool()
            
            ddesi_acc = des.ddesI()
            ddesm_acc = des.ddesM()
            others_acc = des.other_des()

            others_acc.extend([ddesi_acc, ddesm_acc])
            each_result.append(others_acc)
            
        result.append(np.mean(each_result, axis = 0))

    df_result = pd.DataFrame(result, columns = ['knorau', 'knorae', 'desp', 'metades', 'lca', 'ola', 'mcb', 'ddesi', 'ddesm'])
    df_result.to_csv(file_path + data, index = False, header = True)

