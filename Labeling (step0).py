
"""
Created on Sun May 10 01:10:07 2020

@author: Mohammad Mehdi Keramati Feyz Abadi

"""
import pandas as pd
import numpy as np

dish=pd.read_excel('Electricity_DWE.xlsx')
fridge=pd.read_excel('Electricity_FGE.xlsx')
water=pd.read_excel('whole water.xlsx')

    
############################################# Labeling ##################################
counter_1=0
counter_2=0
event=np.zeros((1048575,7))
event=pd.DataFrame(event)

for i in range(1,1048574):
    
    event.loc[i][0]=dish.loc[i]['unix_ts']
    event.loc[i][4]=water.loc[i]['avg_rate']
        
    if (dish.loc[i]['P']-dish.loc[i-1]['P'])>80:
        event.loc[i][1]=dish.loc[i]['P']
    if (dish.loc[i]['P']-dish.loc[i-1]['P'])<-80:
        event.loc[i][1]=dish.loc[i-1]['P']
        
for j in range(1,1048574):
    if (fridge.loc[j]['P']-fridge.loc[j-1]['P'])>80:
        event.loc[j][2]=fridge.loc[j]['P'] 
    if (fridge.loc[j]['P']-fridge.loc[j-1]['P'])<-80:
        event.loc[j][2]=fridge.loc[j-1]['P']
        
        
for k in range(1,1048574):
    if event.loc[k][1]!=0:
        if event.loc[k][2]==0:
            event.loc[k][3]=event.loc[k][1]+event.loc[k][2]
            event.loc[k][5]=1 
            counter_2=counter_2+1
        
    if event.loc[k][1]==0:
        if event.loc[k][2]!=0:
            event.loc[k][3]=event.loc[k][1]+event.loc[k][2]
            counter_2=counter_2+1
            
    if event.loc[k][1]!=0:
        if event.loc[k][2]!=0:
            event.loc[k][3]=event.loc[k][1]
            event.loc[k][5]=1 
            event.loc[k][6]=10101
            counter_1=counter_1+1
            counter_2=counter_2+1
            
event.columns=['time','dish','fridge','total','water','labels','overlap']            
event.to_csv('event_detection.csv')

#####################################################################################
    
        