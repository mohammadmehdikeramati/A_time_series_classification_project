
"""
Created on Sun May 10 01:10:07 2020

@author: Mohammad Mehdi Keramati Feyz Abadi

"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


X=pd.read_csv('balanced_data_ran=26_x.csv',index_col=0)
y=pd.read_csv('balanced_data_ran=26_y.csv',index_col=0)

data_3D=np.zeros((4378,111,2))
for row in range(0,4378):
    for colm in range(0,111):
      data_3D[row][colm][0]=X.loc[2*row][colm]
      data_3D[row][colm][1]=X.loc[2*row+1][colm]
      
############################## Train/Test split and reshape ###################################
      
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(data_3D, y, stratify=y, test_size=0.25)
X_train_1=np.asarray(X_train_1)
X_test_1=np.asarray(X_test_1)
y_train_1=np.asarray(y_train_1)
y_test_1=np.asarray(y_test_1)      
X_train_1=X_train_1.reshape(X_train_1.shape[0], X_train_1.shape[1],2)
X_test_1=X_test_1.reshape(X_test_1.shape[0], X_test_1.shape[1],2)

######################################### Model ##############################################

model = keras.Sequential()
model.add(keras.layers.LSTM(
  units=500,
  input_shape=(X_train_1.shape[1], X_train_1.shape[2]), dropout=0.3, recurrent_dropout=0.0,kernel_initializer='random_uniform'
))
model.add(keras.layers.Dense(units=200,input_shape=(500,1),activation='relu'))

model.add(keras.layers.Dense(units=2,input_shape=(200,1),activation='softmax'))
model.compile(
  loss='binary_crossentropy',
  optimizer=keras.optimizers.Adam(0.001),metrics=['accuracy']
)

le = LabelEncoder().fit(y_train_1)

y_train_1 = np_utils.to_categorical(le.transform(y_train_1), 2)
y_test_1 = np_utils.to_categorical(le.transform(y_test_1), 2)
classTotals = y_train_1.sum(axis=0)


history = model.fit( X_train_1, y_train_1, epochs=50,batch_size=50, validation_split=0.2,verbose=1,shuffle=False)

predictions = model.predict(X_test_1)
pred=pd.DataFrame(predictions)
print(predictions)
print(classification_report(y_test_1.argmax(1),predictions.argmax(1)))


pred.to_csv('prediction.csv')
pd.DataFrame(y_test_1).to_csv('real.csv')
model.save('LSTM.h5') 
#################################################################################################
