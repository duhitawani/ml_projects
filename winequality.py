# -*- coding: utf-8 -*-
"""WineQuality.ipynb
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv("winequality-red.csv")
print(dataset)

from sklearn.model_selection import train_test_split, GridSearchCV

X= dataset.drop('quality',axis = 1)
y= dataset.quality
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2,random_state = 123, stratify = y)
print(X_train)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor
pipeline = make_pipeline(StandardScaler(),RandomForestRegressor(n_estimators = 100))

print(pipeline.get_params)

hyper = {'randomforestregressor__max_features' : ['auto','sqrt','log2'],'randomforestregressor__max_depth':[None,5,3,1]}
gsv = GridSearchCV(pipeline,hyper,cv=10)
gsv.fit(X_train,y_train)
y_pred = gsv.predict(X_test)

from sklearn.metrics import mean_squared_error ,r2_score

print(r2_score(y_test,y_pred))

