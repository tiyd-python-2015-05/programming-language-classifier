from sklearn.pipeline import make_pipeline, make_union
from sklearn.base import TransformerMixin
import re
import itertools


def longest_run_of_capitol_letters_feature(text):
    """Find the longest run of capitol letters and return their length."""
    runs = sorted(re.findall(r"[A-Z]+", text), key=len)
    if runs:
        return [len(runs[-1])]
    else:
        return 0
                                
def percent_character_feature(char_list):
    def feature_fn(text):
        return [text.count(i)/len(text) for i in char_list]
    return feature_fn

def longest_run_of_character_feature(text):
    chars = ['~+', '\.+', '\|+', ';+', '\:+', '\$+', '\(+', '\)+', '\-+']
    runs = []
    for i in chars:
        run = sorted(re.findall(r'{}'.format(i), text), key=len)
    if runs:
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
        """Given a list of original data, return a list of feature vectors."""
        fvs = []
        for datum in X:
            fv = [f(datum) for f in self.featurizers]
            a = list(itertools.chain(*fv))
            fvs.append(a)
        return fvs
