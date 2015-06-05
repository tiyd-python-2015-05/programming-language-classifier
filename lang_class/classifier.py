#!/usr/bin/env python3

from sklearn import cross_validation
from sklearn.feature_selection import VarianceThreshold
from sklearn.externals import joblib
import numpy as np
from scipy.sparse import csc_matrix
import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import os

def main(X_test, Y_test, data=None):
    if X_test is not None:
        X_test = load_matrix(X_test)
    else:
        if data is not None:
            X_test = data
        else:
            raise TypeError('Bad input data')

    classifier = new_classifier()
    prediction = classifier.predict(X_test)
    proba = classifier.predict_proba(X_test)

    if Y_test:

        Y_test = np.load(Y_test)
        result = sum(1 for idx in range(len(Y_test)) if Y_test[idx].lower() != \
                  prediction[idx].lower())

        #for idx in range(len(Y_test)):
        #    print(Y_test[idx].lower(), prediction[idx].lower())

        print('success rate: ', round(100 - 100*result / len(Y_test), 3))
        print('or: ', len(Y_test) - result,' out of ', len(Y_test))

    else:
        keys = np.load('data_keys.npy')
        print(keys[prediction])
        if data is not None:
            for idx in range(len(proba[0])):
                print(keys[idx], round(proba[0][idx], 3))

def new_classifier():
    Xtr = load_matrix('matrix/Xtr.npz')
    Ytr = np.load('matrix/Ytr.npy')
    nb = DecisionTreeClassifier(criterion='entropy') # was MultinomialNB() before
    nb.fit(Xtr, Ytr)

    return nb

def load_matrix(filename):
    filename = filename
    loader = np.load(filename)
    return csc_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a classifier using \
                                      preprocessed files, classify \
                                      input file using the classifier')

    parser.add_argument('--Xtest', default='matrix/Xte.npz', type=str)
    parser.add_argument('--Ytest', default=None, type=str)
    parser.add_argument('--from_text', default=None, type=str)

    args = parser.parse_args()

    data = []
    if args.from_text:
        pipe = joblib.load('dumps/pipe.pkl')
        from_text = args.from_text

        if os.path.isfile(from_text):
            with open(from_text, 'r') as fh:
                data = fh.read()

            data = pipe.transform([data])

        else:
            filenames = os.walk(from_text)
            f = []
            for filename in filenames:
                if filename[1] == []:
                    for item in filename[2]:
                        f.append(filename[0] + '/' + item)

            for item in f:
                with open(item, 'r') as fh:
                    data.append(fh.read())

        data = pipe.transform(data)

        main(None, args.Ytest, data)

    else:
        main(args.Xtest, args.Ytest)
