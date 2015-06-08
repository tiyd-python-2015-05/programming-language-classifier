import re
import itertools
import random
import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import TransformerMixin

import gather_data as gd

def count_characters(text):
    return len(text)

def count_words(text):
    words = [r'\barray\b', r'\bbegin\b', r'\bend\b', r'\bdo\b', r'\bvar\b', r'\bdefn\b', r'\bfunction\b',
            r'\bclass\b', r'\brequire\b', r'\bval\b', r'\bpublic\b', r'\blet\b', r'\bwhere\b', r'\busing\b',
            r'\bextend\b', r'\bfunction\b', r'\bval\b', r'\btry\b']
    results = []
    for word in words:
        results.append(len(re.findall(word, text)))
    return results

def char_runs(text):
    chars = [r'[)]+',r'[}]+', r'[\]]+', r'[=]+']
    results = []
    for char in chars:
        found = sorted(re.findall(char, text), key=len)
        if found:
            results.append(len(found[-1]))
        else:
            results.append(0)
    return results

def percent_characters(text):
    chars = ';!=.<>/\[]{}:_#%$&*'
    results = []
    for char in chars:
        total = max(1, len(text))
        found = text.count(char)
        if found:
            results.append(found / total)
        else:
            results.append(0)
    return results

def endings(text):
    ends = [r'[)]$', r';$', r'}$', r']$', r'\):$']
    results = []
    for end in ends:
        results.append(len(re.findall(end, text, re.MULTILINE)))
    return results


class FunctionFeaturizer(TransformerMixin):
    def __init__(self, *featurizers):
        self.featurizers = featurizers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fvs = []
        for datum in X:
            vec = list(itertools.chain.from_iterable([function(datum) for function in self.featurizers]))
            fvs.append(vec)
        return fvs

class PipelineDebugger(TransformerMixin):
    def __init__(self, name):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(self.name)
        print("=" * 40)
        x = X[random.randrange(0, len(X))]
        print("len:", len(x))
        print(x)
        return X

if __name__ == '__main__':
    texts = []
    tags = []

    languages = ['c', 'perl', 'clojure', 'haskell', 'java', 'javascript', 'ruby', 'ocaml', 'lisp', 'scala', 'csharp', 'php', 'python', 'scheme', 'tcl']
    for language in languages:
        texts.extend(gd.get_code_from_html(language)[0])
        tags.extend(gd.get_code_from_html(language)[1])

    folders = ['clojure', 'csharp', 'gcc', 'hack', 'hs', 'java', 'javascript', 'jruby', 'ocaml', 'perl', 'php', 'python3', 'racket', 'sbcl', 'scala', 'yarv']
    for folder in folders:
        tags.extend(gd.get_benchmark_code(folder)[1])
        texts.extend(gd.get_benchmark_code(folder)[0])

    df_texts = pd.DataFrame(texts)
    print(df_texts.head())
    df_tags = pd.DataFrame(tags)
    merged = pd.merge(df_texts, df_tags, left_index=True, right_index=True)
    merged.columns = ['Snippet', 'Language']

    train_X, test_X, train_y, test_y = train_test_split(merged['Snippet'], merged['Language'], test_size=0.33)

    classifier = Pipeline([('features', FunctionFeaturizer(count_words, percent_characters, char_runs, endings)),
                            ('bayes', MultinomialNB())])
    classifier.fit(train_X, train_y)

    with open("./classifier", "wb") as file:
        pickle.dump(classifier, file)