#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 19:27:48 2020
"""
from yellowbrick.datasets import load_mushroom
from yellowbrick.classifier import ClassificationReport
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


X, y = load_mushroom()

models = [SVC(gamma='auto'),
 NuSVC(gamma='auto'),
 LinearSVC(),
 SGDClassifier(max_iter=100),
 KNeighborsClassifier(),
 LogisticRegression(),
 LogisticRegressionCV(cv=3),
 BaggingClassifier(),
 ExtraTreesClassifier(n_estimators=300),
 GradientBoostingClassifier(n_estimators=300),
 RandomForestClassifier(n_estimators=300)]

estimator_name = []
estimator_score = []

def score_model(X, y, estimator, **kwargs):
    y = LabelEncoder().fit_transform(y)
    model = Pipeline([('one_hot_encoder', OneHotEncoder()), ('estimator', estimator)])
    model.fit(X, y, **kwargs)
    expected = y
    predicted = model.predict(X)
    estimator_name.append(estimator.__class__.__name__)
    estimator_score.append(f1_score(expected, predicted))
    #print(f"({estimator.__class__.__name__}, {f1_score(expected, predicted)})")
    

def visualize_model(X, y, estimator, **kwargs):
    y = LabelEncoder().fit_transform(y)
    
    model = Pipeline([('One_Hot_Encoder', OneHotEncoder()), 
                      ('estimator', estimator)
                     ])
    
    visualizer = ClassificationReport(model, classes=['edible', 'poisonous'], cmap='YlOrRd', support='count')
    visualizer.fit(X, y)
    visualizer.score(X, y)
    visualizer.show()

for model in models:
    score_model(X, y, estimator=model)

score = pd.DataFrame({'name':estimator_name, 'score':estimator_score}).sort_values(by='score', ascending=False)

for model in models:
    visualize_model(X, y, model)


