# -*- coding: utf-8 -*-
"""
Created on Tue May 19 20:00:16 2020

@author: saumya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dt = pd.read_csv('Data.csv')
print(dt)
X = dt.iloc[:, 1:-1].values
y = dt.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X, y , test_size=0.25, random_state=0)


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
reg=KNeighborsClassifier(n_neighbors=5 , metric='minkowski' , p=2)
reg.fit(X_train,y_train)



y_pred=reg.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1) , y_test.reshape(len(y_test),1)) , 1))

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


