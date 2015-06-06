import csv
import re
import numpy as np
import random
import pickle
import itertools
from collections import Counter

from sklearn.pipeline import make_pipeline, make_union
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def longest_run_of_capital_letters_feature(text):
    """Find the longest run of characters and return their length."""
    runs = sorted(re.findall(r"[A-Z]+", text), key=len)
    if runs:
        return [len(runs[-1])]
    else:
        return [0]


def longest_run_of_character_feature(text):
    """Find the longest run of capitol letters and return their length."""
    chars = ["~+", "\.+", "\|+", "\:+", ";+", "\$+", "\(+", "\)+", "\-+", " +", "\t+"]
    runs = []
    for i in chars:
        run = sorted(re.findall(r"{}".format(i), text), key=len)
        if run:
            runs.append(len(run[-1]))
        else:
            runs.append(0)
    return runs


def percent_character_feature(text):
    """Return percentage of text that is a particular char compared to total text length. Must give a list of characters"""
    chars = [".", "|", "$", "_", "!", "#", "@", "%", "^", "&", "*", "(", ")",
             "+", "=", "{", "}", "[", "]", ":", ";", "?", "<", ">"]
    return [text.count(i)/len(text) for i in chars]


def percent_character_combinations(text):
    """Return percentage of text that is a particular char compared to total text length. Must give a list of characters"""
    chars = ["==", "\->+", ":\-+", "\+=", "\n\t+if", "\n+", "\n\$+", "\n\t+", "\ndef", "%{", "~=", "\|\|",
             "\n\t+\(\w+", "^\$", "\.=", "\{:", "===", "!==", "\*\w+", "__", "__name__", "__main__", "^\#"
         "^def", "^@w+", "^@end", "^begin", "^end", "^functions", "^loop\n", "^procedure", "^func",
         "\+\+"]
    runs = []
    for i in chars:
        run = re.findall(r"{}".format(i), text)
        if run:
            runs.append(len(run)/len(text))
        else:
            runs.append(0)
    return runs

def binary_character_combinations(text):
    """Return binary of text that is a particular char compared to total text length. Must give a list of characters"""
    chars = ["==", "\->+", ":\-+", "\+=", "\n\t+if", "\n+", "\n\$+", "\n\t+", "\ndef", "%{", "~=", "\|\|",
             "\n\t+\(\w+", "^\$", "\.=", "\{:", "===", "!==", "\*\w+", "__", "__name__", "__main__", "^\#"
         "^def", "^@w+", "^@end", "^begin", "^end", "^functions", "^loop\n", "^procedure", "^func",
         "\+\+"]
    runs = []
    for i in chars:
        run = re.findall(r"{}".format(i), text)
        if run:
            runs.append(1)
        else:
            runs.append(0)
    return runs


def make_pipe(estimator):
    """make_pipe function must have the type of estimator e.g RandomForestClassifier()"""
    language_featurizer = make_union(CountVectorizer(),
                                     FunctionFeaturizer(longest_run_of_capital_letters_feature,
                                                        longest_run_of_character_feature,
                                                        percent_character_combinations,
                                                        percent_character_feature,
                                                        binary_character_combinations))

    return make_pipeline(language_featurizer, estimator)

class FunctionFeaturizer(TransformerMixin):
    def __init__(self, *featurizers):
        self.featurizers = featurizers

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        return self


    def transform(self, X):
        fvs = []
        for datum in X:
            fv = [f(datum) for f in self.featurizers]
            a = list(itertools.chain(*fv))
            fvs.append(a)
        # b = list(itertools.chain(*fvs))
        return fvs