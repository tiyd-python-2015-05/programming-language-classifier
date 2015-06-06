import re
import glob

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
import os.path

# TODO: Future Ideas:
# use n-grams?
# inverse_transform or otherwise make an best-case exemplar
# web scraping

def make_extension_dict():
    extensions = {'C': ['gcc', 'c'],
                  'C#': 'csharp',
                  'Common Lisp': 'sbcl',
                  'Clojure': 'clojure',
                  'Haskell': 'haskell',
                  'Java': 'java',
                  'JavaScript': 'javascript',
                  'OCaml': 'ocaml',
                  'Perl': 'perl',
                  'PHP': ['hack', 'php'],
                  'Python': 'python3',
                  'Ruby': ['jruby', 'yarv'],
                  'Scala': 'scala',
                  'Scheme': 'racket',
                  }

    ext_lookup = {}
    for key, value in extensions.items():
        """Flip the dictionary around"""
        if type(value) == type([]):  # hasattr(value, '__iter__'):
            for value2 in value:
                ext_lookup[value2] = key
        else:
            ext_lookup[value] = key
    return ext_lookup


def extract_extension(string):
    match = re.match('.*\.(?P<ext>.*)$', string)
    if match:
        return match.groupdict()['ext']


def load_bench_data(reload=False):
    df = unpickle('bench.data', reload=reload)
    # if os.path.isfile("bench.data") and not reload:
    #     df = pickle.load( open( "bench.data", "rb" ) )
    #     return df
    if df is not None:
        return df
    df = pd.DataFrame(columns=['language', 'text'])
    files = glob.glob('bench/*/*')

    for fn in files:
        try:
            with open(fn) as fh:
                data = {'language': extract_extension(fn),
                        'text': ''.join(fh.readlines())}
                if data['language'] and data['text']:
                    df = df.append(data, ignore_index=True)
        except (IsADirectoryError, UnicodeDecodeError):
            pass
    pickle.dump(df, open("bench.data", "wb"))
    return df


def load_test_data():
    test_data = pd.read_csv('./test.csv', names=['item', 'language', 'text', 'guess'])
    test_files = glob.glob('./test/*')

    for (idx, fn) in enumerate(test_files):
        #     try:
        with open(fn) as fh:
            #         df.loc[extract_extension(fn)] = ''.join(fh.readlines())
            #         data = {'language': extract_extension(fn),
            #                 'text': ''.join(fh.readlines())}
            #         if data['language'] and data['text']:
            #             df = df.append(data, ignore_index = True)
            #     except (IsADirectoryError, UnicodeDecodeError):
            #         pass
            # test_data['text'][idx] = ''.join(fh.readlines())
            test_data.ix[idx, 'text'] = ''.join(fh.readlines())
    return test_data


def assess_classifier(pipe, *split_args):
    pipe.fit(split_args[0], split_args[2])
    train_score = pipe.score(split_args[0], split_args[2])
    test_score = pipe.score(split_args[1], split_args[3])
    print('Train score: {:.3f}, Test score: {:.3f}'.format(train_score, test_score))
    return pipe


def longest_run_of_caps_feature(text):
    runs = sorted(re.findall(r"[A-Z]+", text), key=len)
    if len(runs) == 0:
        return [0]
    longest = runs[-1]
    return [len(longest)]


def percent_periods_feature(text):
    """Return percentage of text that is periods compared to total text length."""
    periods = text.count(".")
    return [periods / len(text)]


def unpickle(name, reload=False):
    if os.path.isfile(name) and not reload:
        df = pickle.load(open("bench.data", "rb"))
        return df
    else:
        return None


class CustomFeaturizer:
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
            fv = np.array([f(datum) for f in self.featurizers])
            fvs.append(fv.reshape(1, -1)[0])
        return np.array(fvs)


if __name__ == '__main__':
    df = load_bench_data()
    X = df.text
    y = df.language
    test_data = load_test_data()

    args = train_test_split(X, y, test_size=0.2, )  # random_state=0) # X_train, X_test, y_train, y_test

    spam_pipe = Pipeline([('bag_of_words', CountVectorizer()),
                          ('bayes', MultinomialNB())])
    print(spam_pipe)
    classifier = assess_classifier(spam_pipe, *args)
    classifier.predict(args[1])

    spam_pipe = Pipeline([('bag_of_words', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('RFC', RandomForestClassifier())])
    spam_pipe.set_params(RFC__n_estimators=1000)
    print(spam_pipe)
    classifier = assess_classifier(spam_pipe, *args)

    test_data['guess'] = pd.DataFrame(spam_pipe.predict(test_data['text']))
    correct = test_data[test_data.language == test_data.guess]
    print('Proportion of test data correctly labeled: {:.3f}'.format(len(correct) / len(test_data)))
    print(test_data)


    featurizer = CustomFeaturizer(longest_run_of_capitol_letters_feature,
                                  percent_periods_feature)
