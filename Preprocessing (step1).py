
"""
Created on Sun May 10 01:10:07 2020

@author: Mohammad Mehdi Keramati Feyz Abadi

"""

import numpy as np
import pandas as pd


data=pd.read_csv('event_detection.csv',index_col=0)

data_c=data.copy()
data_c.index=data_c.time
data_c=data_c.drop('time',axis=1)

####################################### Windows Creation ###################################

time_steps = 110 
def create_dataset(X,z,y, time_steps=1):  
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v_1 = X.iloc[i:(i + time_steps+1)].values
        Xs.append(v_1)
        v_2 = z.iloc[i:(i + time_steps+1)].values
        Xs.append(v_2)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

#############################################################################################

########################### Class Blancing (Random Overampling) #############################



X_train_1, y_train_1 = create_dataset(data_c.total,data_c.water,data_c.labels, time_steps)


X_train=pd.DataFrame(X_train_1)
y_train=pd.DataFrame(y_train_1)

X_train.append(y_train)
X_train.to_csv('X_train_1.csv')
y_train.to_csv('y_train_1.csv')

one=0
for z in range (0,1048464):
    if y_train.loc[z][0]!=0:
        if X_train.loc[2*z][110]<200:
          if X_train.loc[2*z][110]!=0:
            one=one+1
         
zero=0
for j in range (0,1048464):
    if y_train.loc[j][0]==0:
      if X_train.loc[2*j][110]<200:
          if X_train.loc[2*j][110]!=0:
            zero=zero+1
      
    

data_one=np.zeros((2*one,111))
data_zero=np.zeros((2*zero,111))



m=0

for i in range (0,1048464):
    
    if y_train.loc[i][0]==1:
      if X_train.loc[2*i][110]!=0:
        if X_train.loc[2*i][110]<200:
          data_one[2*m][:]=X_train.loc[2*i][:]
          data_one[2*m+1][:]=X_train.loc[2*i+1][:]
          m=m+1
        
m=0

for q in range (0,1048464):
           
    if y_train.loc[q][0]==0:
      if X_train.loc[2*q][110]!=0:
        if X_train.loc[2*q][110]<200:
         data_zero[2*m][:]=X_train.loc[2*q][:]
         data_zero[2*m+1][:]=X_train.loc[2*q+1][:]
         m=m+1
        
        
data_c=np.zeros((4*one,111))
data_d=np.zeros((2*one,1))
data_one=pd.DataFrame(data_one)
data_zero=pd.DataFrame(data_zero)


m=0        
for p in range (0,one):
    data_c[4*m][:]=data_one.loc[2*p][:]
    data_c[4*m+1][:]=data_one.loc[2*p+1][:]
    data_d[2*m][0]=1
    data_c[4*m+2][:]=data_zero.loc[26*p][:]
    data_c[4*m+3][:]=data_zero.loc[26*p+1][:]
    m=m+1

data_c=pd.DataFrame(data_c) 
data_d=pd.DataFrame(data_d)
data_c.to_csv('balanced_data_ran=26_x.csv')
data_d.to_csv('balanced_data_ran=26_y.csv')

#########################################################################################





