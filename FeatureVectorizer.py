from sklearn.pipeline import make_pipeline, make_union
from sklearn.base import TransformerMixin
import re
import itertools


def longest_run_of_capital_letters_feature(text):
    """Find the longest run of capital letters and return their length."""
    runs = sorted(re.findall(r"[A-Z]+", text), key=len)
    if runs:
        return [len(runs[-1])]
    else:
        return [0]

    
def longest_run_of_character_feature(text):
    chars = ['~+', '\.+', '\|+', ';+', '\:+', ';+', '\$+', '\(+', '\)+', '\-+', '\s+', '\t+']
    runs = []
    for i in chars:
        run = sorted(re.findall(r'{}'.format(i), text), key=len)
    if runs:
        runs.append(len(run[-1]))
    else:
        runs.append(0)
    return runs


def percent_character_feature(text):
    """Return percentage of text that is a particular char compared to total text length."""
    chars = [".", "|", "$", "_", "!", "#", "@", "%", "^", "&", "*", "(", ")","+", "=", "{", "}", "[", "]", ":", ";", "?", "<", ">"]

    return [text.count(i)/len(text) for i in chars]


def percent_character_combinations(text):
    """Return percentage of text that is a particular char compared to total text length."""
    chars = ["==", "\->+", ":\-+", "\+=", "\n\t+if", "\n+", "\n\$+", "\n\t+", "\ndef", "%{", "~=", "\|\|", "\n\t+\(\w+", "^\$", "\.=", "\{:", "===", "!==", "\*\w+", "__", "__name__", "__main__", "^\#", "^def", "^@w+", "^@end", "^begin", "^end", "^functions", "^loop\n", "^procedure", "^func","\+\+"]
    runs = []
    for i in chars:
        run = re.findall(r'{}'.format(i), text)
        if run:
            runs.append(len(run)/len(text))
        else:
            runs.append(0)
    return runs

def binary_character_combinations(text):
    '''Return binary of text that is particular char to total length of text'''
    chars = ["==", "\->+", ":\-+", "\+=", "\n\t+if", "\n+", "\n\$+", "\n\t+", "\ndef", "%{", "~=", "\|\|","\n\t+\(\w+", "^\$", "\.=", "\{:", "===", "!==", "\*\w+", "__", "__name__", "__main__", "^\#", "^def", "^@w+", "^@end", "^begin", "^end", "^functions", "^loop\n", "^procedure", "^func","\+\+"]
    runs = []
    for i in chars:
        run = re.findall(r'{}'.format(i), text)
        if run:
            runs.append(1)
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
