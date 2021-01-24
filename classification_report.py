#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from yellowbrick.datasets import load_occupancy
from yellowbrick.classifier import classification_report

df = load_occupancy()

for each in df.head():
    print(each)


