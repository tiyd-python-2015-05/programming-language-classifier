import csv
import re
import numpy as np
import random

#from textblob import TextBlob
from collections import Counter

from sklearn.pipeline import make_pipeline, make_union
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class DumbFeaturizer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        matrix = []
        for i in range(len(X)):
            vector = []
            for j in range(11):
                if j == X[i]:
                    vector.append(1)
                else:
                    vector.append(0)
            matrix.append(vector)
        return matrix

N = 22
y = [0] * N
X = [0] * N
for k in range(N):
    val = random.randrange(11)
    y[k] = val
    X[k] = val


dumb = DumbFeaturizer()
print(dumb.transform(X))

pipe = make_pipeline(dumb, DecisionTreeClassifier())
pipe.fit(X, y)
# Our baseline
print(pipe.score(X, y))
print(" ")
print(" transform ")
print(pipe.transform(X))
