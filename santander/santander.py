# -*- coding: utf-8 -*-
"""
Created on Mon May  2 11:39:03 2016

@author: naamas
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series,DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn import metrics as met

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFpr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection  import chi2
from sklearn.feature_selection  import f_classif
from sklearn.feature_selection  import SelectPercentile
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier 

class Data():
    
    def __init__(self):
        
        self.X_train=[]
        self.X_test=[]
        self.y_train=[]
        self.y_test=[]
        
    def setdata(self,X_train,y_train,X_test,y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        
class DataCreate():

    def __init__(self, pathTO, submit=0):
       
        self.real_test_x=[]
        self.path=pathTO
     
        self.columns=[]
        self.submit=submit
    
    def read_file(self):
        '''
        Outputs: X- the csv as a pandas dataframe
        Description: read a csv file with a given path and return the data as a pandas df  
        '''
        
        X= pd.read_csv(self.path)
        if(self.submit):
          self.real_test_x= pd.read_csv('C:\\Temp\\test.csv')
        return X 
        
        
    ##############################
    # Name: divide_data
    # Inputs: self, X- a pandas df ,train size , test size, data class object, seed for random generation
    # Outputs: none 
    # Description:  divieds the data to train/test groups, remove ID and TARGET and set the y train and test 
    ##############################    
    def divide_data(self,X,train_size,test_size,data,seed):  
        
        data.X_train, data.X_test= train_test_split(X,train_size=train_size,test_size=test_size,random_state=seed)        
        self.create_y("TARGET",data)
        self.delete_features( "TARGET",data)
        self.delete_features( "ID",data)
    
    ##############################
    # Name: create_y
    # Inputs: self, y_name -the colums name that contains y and  data class object
    # Outputs: none 
    # Description:  sets the y train and test 
    ##############################     
    
    def create_y(self,y_name,data):
            data.y_train=data.X_train[y_name]
            if(self.submit):   
                data.y_test=0
            else:
                data.y_test=data.X_test[y_name] 
                
    ##############################
    # Name: delete_features
    # Inputs: self,feature_name,data tada object
    # Outputs: none 
    # Description:  remove a feature fro the test and train sets (if exists)
    ##############################                
    def delete_features(self, feature_name,data):
        if any(col==feature_name for col in data.X_train.columns): 
            data.X_train=data.X_train.drop([feature_name],axis=1)
        if any(col==feature_name for col in data.X_test.columns):     
            data.X_test=data.X_test.drop([feature_name],axis=1) 
    

class PreprocessData():
    
    def __init__(self, data):
        
        self.data=data
         
    def change_values_of_feature(self,value,new_value,feature_name):
    
        self.data.X_train[feature_name].replace(value,new_value,inplace=True)
        self.data.X_test[feature_name].replace(value,new_value,inplace=True)
      
        
    def univariant_feature_selection(self,method, X, y,percentile):
        
        test=SelectPercentile(method , percentile=percentile).fit(X, y)
        print("The number of feature in ", method, " is: ", (test.get_support().sum()) )
        for i in range(len(self.X_train.columns)):
            if(test.get_support()[i]):
                print(self.X_train.columns[i])
        return  test.get_support()  
        
    def pca_feature_selection(self,num_of_features_to_keep):
        
        pca = PCA(n_components=num_of_features_to_keep)
        train_pca=pca.fit_transform(normalize(self.data.X_train,axis=0),self.data.y_train )
        test_pca=pca.transform(normalize(self.data.X_test))
        return ([train_pca,test_pca])
        
    def add_pca_features(self,num_of_features_to_keep,pca_train_features,pca_test_features):
        
        for i in range(num_of_features_to_keep):
            feature_name="pca%d"  % i
            self.add_features(pca_train_features[:,i],feature_name ,self.data.X_train)
            self.add_features(pca_test_features[:,i],feature_name,self.data.X_test)
            
    def add_features(self,feature_to_add,feature_name,df):
        
        df[feature_name]=feature_to_add
        
    def clean_data(self):
       
        ind_arr=self.find_non_const_ind()
        ind_arr=self.remove_equal_features(ind_arr)
        self.build_a_new_mat(ind_arr)
        
        return ind_arr   
         
    def feature_selection(self,percentile):
        final_arr=[]
        ind_arr2=[]
           
        f_test=self.univariant_feature_selection(f_classif, self.data.X_train, self.data.y_train,percentile)
        chi2_test=self.univariant_feature_selection(chi2, Binarizer().fit_transform(scale(self.data.X_train)), self.data.y_train,percentile)
        or_arr=f_test|chi2_test
        for i in range(len(self.data.X_train.columns)):
            if(or_arr[i]):
                ind_arr2.append(i)
        final_arr=self.find_correlated_features(ind_arr2,0.99)        
        return final_arr
    
    def build_a_new_mat(self,ind_arr):
        self.data.X_train=data.X_train[ind_arr]
        print(self.data.X_train.shape)
        print(self.data.y_train.shape)  
        self.data.X_test=self.data.X_test[ind_arr]
        print(self.data.X_test.shape)
        
                    
    def find_non_const_ind(self):
        
        ind_arr=[] 
        columns = self.data.X_train.columns
        for i in range(self.data.X_train.shape[1]):
            if(self.data.X_train[columns[i]].std()>0):
                ind_arr.append(i)
        return ind_arr 

    def find_correlated_features(self,ind_arr,threshold):
     
        arr_to_del=[]
     
        columns = self.data.X_train.columns

        for i in range(len(ind_arr)-1):
            for j in range(i+1,len(ind_arr)):      
                if(self.data.X_train[columns[ind_arr[i]]].corr(self.data.X_train[columns[ind_arr[j]]])>threshold):
                    arr_to_del.append(ind_arr[j])
       
        tmp_arr=np.unique(arr_to_del) 
        for i in range(len(tmp_arr)):
            ind_arr.remove(tmp_arr[i])
                     
        return ind_arr 
        
    def remove_equal_features(self,ind_arr):
     
        arr_to_del=[]
     
        columns = self.data.X_train.columns

        for i in range(len(ind_arr)-1):
            for j in range(i+1,len(ind_arr)):      
                if(np.array_equal(self.data.X_train[columns[ind_arr[i]]].values,self.data.X_train[columns[ind_arr[j]]].values)):
                    arr_to_del.append(ind_arr[j])
       
        tmp_arr=np.unique(arr_to_del) 
        for i in range(len(tmp_arr)):
            ind_arr.remove(tmp_arr[i])
                     
        return ind_arr  
        
    def find_non_sparse_ind1(self,percent):
        ind_arr=[]
        num_of_non_sparse=(percent*len(self.data.y_train.nonzero()[0]))
        nz_ind=self.data.y_train.nonzero()     
     
        columns = self.data.X_train.columns

        for i in range(self.data.X_train.shape[1]):
             if((self.data.X_train.iloc[nz_ind][columns[i]].nonzero()[0].shape[0])>=num_of_non_sparse):
                 ind_arr.append(i)
        return ind_arr 
        
        
class ManageLearning():
    
    def __init__(self,data,check_flag):
        
        self.res_matrix_train=DataFrame()
        self.res_matrix_test=DataFrame()
        self.data=data
        self.check_flag=check_flag
        
    def learn_model(self,model,model_name,probabilty_flag):
        learn=Learn(model)
        [self.res_matrix_test[model_name],self.res_matrix_train[model_name]]=learn.classify(self.data,probabilty_flag,self.check_flag)
        return self.res_matrix_test[model_name]
        
    def combine_results(self,model,probabilty_flag,check_flag):
        learn=Learn(model)
        y_test_pred=learn.classify(self.res_matrix_train,self.data,probabilty_flag,self.check_flag)[0]
        return y_test_pred
        
    def submission(self,y_test_pred,real_test_x):
    
        sub = DataFrame()
        sub["ID"] = real_test_x["ID"]
        sub["TARGET"] = y_test_pred
        sub.to_csv('C:\\Temp\\santander.csv', index=False)
    
class Learn():
    
    def __init__(self, model = KNeighborsClassifier):
        
        self.most_important_features=[]
       
        self.model = model

    def __plot_auc(self,arr_true, arr_predict):
        
        fpr, tpr, _ = met.roc_curve(arr_true, arr_predict)
        roc_auc = met.auc(fpr, tpr)
        print ("the auc is: :",roc_auc )
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    
    def calc_auc(self,arr_true, arr_predict):
        
        fpr, tpr, _ = met.roc_curve(arr_true, arr_predict)
        roc_auc = met.auc(fpr, tpr)
        print ("the auc is: :",roc_auc )
        return roc_auc
        
    def __check_results(self,arr_predict, arr_true):
        
        label_pred= (arr_predict+0.5).astype(int)
        print(met.classification_report(arr_true,label_pred))
        print(met.confusion_matrix(arr_true, label_pred))
        self.__plot_auc(arr_true, arr_predict)
        
    def classify(self,data,probabilty_flag,check_flag):
           
        self.model.fit(data.X_train, data.y_train)
        if(probabilty_flag):
            y_test_pred = self.model.predict_proba(data.X_test)[:,1]
            y_train_pred=self.model.predict_proba(data.X_train)[:,1]
        
        else:
            y_test_pred = self.model.predict(data.X_test)[:,1]
            y_train_pred =self.model.predict(data.X_train)[:,1]
            
        if(check_flag==1):
             self.__check_results(y_test_pred, data.y_test)  
             
        return [y_test_pred,y_train_pred]  
        
          
            
if __name__ == '__main__':       
    import pdb
    pdb.set_trace()        
    submit=1
    check_flag=1-submit
    probability_flag=1    
    p='C:\\Temp\\train.csv'
    data=Data(p,submit)
    data.divide_data(data.read_file(),75000,0)
    data.change_values_of_feature(-999999,2.71,"var3")
 
    
    num_of_iterations=5
    auc_arr=[]
    
    
    data.clean_data()
    [train_pca,test_pca]=data.pca_feature_selection(2)
    data.add_pca_features(2,train_pca,test_pca)
    #for i in range(num_of_iterations):
       
    ind=data.feature_selection(25+i)
      
        
         ###add pca to ind
        
     #   ind.append(data.X_train.shape[1]-1)
     #   ind.append(data.X_train.shape[1]-2)
     #use_columns=data.X_train.columns[np.unique(ind)]
    mng=ManageLearning(data.X_train[use_columns],data.X_test[use_columns],data.y_train,data.y_test,check_flag)
        #mng.learn_model(GradientBoostingClassifier(n_estimators=400, learning_rate=0.03,max_depth=5, random_state=0),"grad_boost",probability_flag)
    pred=mng.learn_model(AdaBoostClassifier(n_estimators=400, learning_rate=0.03),"ada_boost",probability_flag)
    auc_arr.append(Learn.calc_auc(data.y_test, pred))
       # y_test_pred=mng.combine_results(LogisticRegression(class_weight={0:0.2, 1:0.8}),probability_flag,check_flag)
    
    #learn.learn_logistic(check_flag)
    #learn.learn_adaboost(check_flag)
    #feature exploration 
    print (auc_arr)
   #learn.learn_gradboost(check_flag)
    #learn.combine_results(check_flag)
    if(submit):
        mng.submission(y_test_pred,data.real_test_x)
    
 
    #df = pd.read_csv(p)
    #print(df.dtypes)
    #df_new=df.drop('ID',1)
    #df_new.columns=[np.arange(370)]
    #pd.Series(df_new[[0]]).nonzero()
