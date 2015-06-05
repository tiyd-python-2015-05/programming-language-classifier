import re
import numpy as np
import random
from collections import Counter
import itertools
from sklearn.pipeline import make_pipeline, make_union
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def longest_run_of_capitol_letters_feature(text):
    """Find the longest run of capitol letters and return their length."""
    runs = sorted(re.findall(r"[A-Z]+", text), key=len)
    if runs:
        return [len(runs[-1])]
    else:
        return [0]


def percent_character_feature(char_list):
    """Return percentage of text that is a particular char compared to total text length."""
    def feature_fn(text):
        return [text.count(i)/len(text) for i in char_list]
    return feature_fn


def longest_run_of_tabs_feature(text):
    """Find the longest run of capitol letters and return their length."""
    runs = sorted(re.findall(r"\t+", text), key=len)
    if runs:
        return [len(runs[-1])]
    else:
        return [0]


def longest_run_of_spaces_feature(text):
    """Find the longest run of capitol letters and return their length."""
    runs = sorted(re.findall(r" +", text), key=len)
    if runs:
        return [len(runs[-1])]
    else:
        return [0]


def longest_run_of_character_feature(text):
    """Find the longest run of char and return length."""
    chars = ["~+", "\.+", "\|+", ";+", "\:+", "\$+", "\(+", "\)+", "\-+"]
    runs = []
    for i in chars:
        run = sorted(re.findall(r"{}".format(i), text), key=len)
        if run:
            runs.append(len(run[-1]))
        else:
            runs.append(0)
    return runs


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
        return fvs

#
#
# sms_featurizer = make_union(
#     CountVectorizer(),
#     FunctionFeaturizer(longest_run_of_capitol_letters_feature,
#                        percent_character_feature(".")))
#
#
# pipe = make_pipeline(sms_featurizer, DecisionTreeClassifier())
# pipe.fit(X_train, y_train)
# pipe.score(X_test, y_test)