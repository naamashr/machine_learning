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
#import xgboost as xgb

class Data(object):

    def __init__(self, pathTO,submit=0):
        self.X = []
        self.y =[]
        self.real_test_x=[]
        self.path=pathTO
        self.X_train=[]
        self.X_test=[]
        self.y_train=[]
        self.y_test=[]
        self.work_X_train=[]
        self.work_X_test=[]
        self.work_y_train=[]
        self.work_y_test=[]
        self.columns=[]
        self.submit=submit
    
    def read_file(self):
        self.X= pd.read_csv(self.path)
        if(self.submit):
          self.real_test_x= pd.read_csv('C:\\Temp\\test.csv')

    def divide_data(self,train_size,test_size):  
        
        self.X_train, self.X_test= train_test_split(self.X,train_size=train_size,test_size=test_size)        
        
#        self.X_train=self.X[:][0:train_size]
        self.y_train=self.X_train['TARGET']
        self.X_train=self.X_train.drop(["ID","TARGET"],axis=1)

        if(self.submit):
            self.X_test=self.real_test_x
            self.y_test=0
            self.X_test=self.X_test.drop(["ID"],axis=1) 

        else:
#            self.X_test=self.X[:][train_size:(train_size+test_size)]
            self.y_test=self.X_test['TARGET'] 
            self.X_test=self.X_test.drop(["ID","TARGET"],axis=1) 
            
    def univariant_feature_selection(self,method, X, y,percentile):
        test=SelectPercentile(method , percentile=percentile).fit(X, y)
        print("The number of feature in ", method, " is: ", (test.get_support().sum()) )
        for i in range(len(self.X_train.columns)):
            if(test.get_support()[i]):
                print(self.X_train.columns[i])
        return  test.get_support()    
        
    def feature_selection(self,percentile,skip):
        final_arr=[]
        ind_arr2=[]
        ind_arr=self.find_non_const_ind()
        ###in var 3 replace -999999 in the mean value 2.71
        self.X_train.var3.replace(-999999,2.71)
        self.X_test.var3.replace(-999999,2.71)
        if(self.submit):
          self.real_test_x.var3.replace(-999999,2.71)
        if(skip==1):
            return ind_arr2
        self.build_a_new_mat(ind_arr)
        f_test=self.univariant_feature_selection(f_classif, self.X_train, self.y_train,percentile)
        chi2_test=self.univariant_feature_selection(chi2, Binarizer().fit_transform(scale(self.X_train)), self.y_train,percentile)
        or_arr=f_test|chi2_test
        for i in range(len(self.X_train.columns)):
            if(or_arr[i]):
                ind_arr2.append(i)
        final_arr=self.find_correlated_features(ind_arr2,0.99)        
        return final_arr
    
    def build_a_new_mat(self,ind_arr):
        self.X_train=self.X_train[ind_arr]
        print(self.X_train.shape)
        print(self.y_train.shape)  
        self.X_test=self.X_test[ind_arr]
        print(self.X_test.shape)
        if(self.submit==0):
            print(self.y_test.shape)     
    
    def find_non_const_ind(self):
     ind_arr=[] 
     columns = self.X_train.columns
     for i in range(self.X_train.shape[1]):
         if(self.X_train[columns[i]].std()>0):
             ind_arr.append(i)
     return ind_arr 


    def find_correlated_features(self,ind_arr,threshold):
     
     arr_to_del=[]
     
     columns = self.X_train.columns

     for i in range(len(ind_arr)-1):
         for j in range(i+1,len(ind_arr)):      
             if(self.X_train[columns[ind_arr[i]]].corr(self.X_train[columns[ind_arr[j]]])>threshold):
                 arr_to_del.append(ind_arr[j])
   
     tmp_arr=np.unique(arr_to_del) 
     for i in range(len(tmp_arr)):
         ind_arr.remove(tmp_arr[i])
                 
     return ind_arr 
     
     
     
    def find_non_sparse_ind1(self,percent):
        ind_arr=[]
        num_of_non_sparse=(percent*len(self.y_train.nonzero()[0]))
        nz_ind=self.y_train.nonzero()     
     
        columns = self.X_train.columns

        for i in range(self.X_train.shape[1]):
             if((self.X_train.iloc[nz_ind][columns[i]].nonzero()[0].shape[0])>=num_of_non_sparse):
                 ind_arr.append(i)
        return ind_arr 


class Learn(object):
    def __init__(self):
        
        self.y_test_pred=[]
        self.most_important_features=[]
        self.res_matrix_train=DataFrame()
        self.res_matrix_test=DataFrame()

    def nCr(self,n,k):
        f = math.factorial
        return f(n) / f(k) / f(n-k)

    def plot_auc(self,arr_true, arr_predict):
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
    
   
    def check_results(self,arr_predict, arr_true):
        label_pred= (arr_predict+0.5).astype(int)
        print(met.classification_report(arr_true,label_pred))
        print(met.confusion_matrix(arr_true, label_pred))
        #calculateing aree under curve:
        self.plot_auc(arr_true, arr_predict)
    
       
    def learn_knn(self):
        from sklearn.neighbors import KNeighborsClassifier    
        model = KNeighborsClassifier()
        model.fit(data.X_train, data.y_train)
        self.y_test_pred=model.predict(data.X_test)
        self.res_matrix_test["knn"]=self.y_test_pred[:,1]
        self.res_matrix_train["knn"]=model.predict_proba(data.X_train)[:,1]
        print(self.y_test_pred)
        print(data.y_test)

    def  learn_logistic(self,check_flag):
        
         model=LogisticRegression(class_weight={0:0.2, 1:0.8}) 
         model.fit(data.X_train, data.y_train)
         self.y_test_pred=model.predict_proba(data.X_test)
         self.res_matrix_test["logistic"]=self.y_test_pred[:,1]
         self.res_matrix_train["logistic"]=model.predict_proba(data.X_train)[:,1]
         if(check_flag==1):
             self.check_results(self.y_test_pred[:,1], data.y_test)
           
             
    def learn_adaboost(self,check_flag):
          model = AdaBoostClassifier(
                         learning_rate=0.1,
                         n_estimators=500)
          model.fit(data.X_train, data.y_train)               
          self.y_test_pred=model.predict_proba(data.X_test)
          self.most_important_features=model.feature_importances_.argsort()
          self.res_matrix_test["adaboost"]=self.y_test_pred[:,1]
          self.res_matrix_train["adaboost"]=model.predict_proba(data.X_train)[:,1]
          if(check_flag==1):
              self.check_results(self.y_test_pred[:,1], data.y_test)
                
    def learn_gradboost(self,check_flag):
          model = GradientBoostingClassifier(n_estimators=400, learning_rate=0.03,max_depth=5, random_state=0)

          model.fit(data.X_train, data.y_train)               
          self.y_test_pred=model.predict_proba(data.X_test)
          self.res_matrix_test["gradboost"]=self.y_test_pred[:,1]
          self.res_matrix_train["gradboost"]=model.predict_proba(data.X_train)[:,1]
          if(check_flag==1):
              self.check_results(self.y_test_pred[:,1], data.y_test)          
          
          
    def combine_results(self,check_flag):
        logisticreg=LogisticRegression(class_weight={0:0.2, 1:0.8})
        logisticreg.fit(self.res_matrix_train, data.y_train)
        self.y_test_pred=logisticreg.predict_proba(self.res_matrix_test)
        if(check_flag==1):
            self.check_results(self.y_test_pred[:,1], data.y_test)
            
    def submission(self):
        
        sub = DataFrame()
        sub["ID"] = data.real_test_x["ID"]
        sub["TARGET"] = self.y_test_pred

        sub.to_csv('C:\\Temp\\santander.csv', index=False)

         
import pdb
pdb.set_trace()        
submit=0  
check_flag=1-submit    
p='C:\\Temp\\train.csv'
data=Data(p,submit)
data.read_file()
data.divide_data(25000,25000)
#ind=data.find_non_sparse_ind1(0.02)
ind=data.feature_selection(10,0)
data.build_a_new_mat(ind)
learn=Learn()
#learn.learn_logistic(check_flag)
#learn.learn_adaboost(check_flag)
#feature exploration 

learn.learn_gradboost(check_flag)
#learn.combine_results(check_flag)
if(submit):
    learn.submission()
    
 
#df = pd.read_csv(p)
#print(df.dtypes)
#df_new=df.drop('ID',1)
#df_new.columns=[np.arange(370)]
#pd.Series(df_new[[0]]).nonzero()
