#!/usr/bin/env python3

from sklearn import cross_validation
from sklearn.externals import joblib
import numpy as np
from scipy.sparse import csc_matrix
import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os


def main(X_test, Y_test, data=None, model='dt'):
    """
    Creates a classifier, trains the classifier
    on preprocessed data, then predicts X_test
    data.  If Y_test data given, returns the
    success rate and counts, otherwise
    returns the predictions
    """

    if X_test is not None:
        X_test = load_matrix(X_test)
    else:
        if data is not None:
            X_test = data
        else:
            raise TypeError('Bad input data')

    classifier = new_classifier(model)
    prediction = classifier.predict(X_test)

    proba = classifier.predict_proba(X_test)

    if Y_test:

        Y_test = np.load(Y_test)
        result = sum(1 for idx in range(len(Y_test)) if Y_test[idx] != \
                  prediction[idx])

        print('success rate: ', round(100 - 100*result / len(Y_test), 3))
        print('or: ', len(Y_test) - result,' out of ', len(Y_test))

    else:
        keys = np.load('data_keys.npy')

        for item in prediction:
            print(keys[item])


        if data is not None:
            for idx in range(len(proba[0])):
                print(keys[idx], round(proba[0][idx], 3))

def new_classifier(model):
    """
    creates a classifier, either Decision Tree or linear
    trains it on the preprocessed training data
    and returns the classifier
    """
    Xtr = load_matrix('matrix/Xtr.npz')
    Ytr = np.load('matrix/Ytr.npy')
    if model == 'dt':
        nb = DecisionTreeClassifier(criterion='entropy')
    if model == 'bayes':
        nb = MultinomialNB()

    nb.fit(Xtr, Ytr)

    return nb

def load_matrix(filename):
    """
    imports the preprocessed training data
    """
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

    parser.add_argument('-bayes', help='Use Multinomial Naive Bayes',
                        action='store_true')


    args = parser.parse_args()

    if args.bayes:
        model = 'bayes'
    else:
        model = 'dt'

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

        main(None, args.Ytest, data, model)

    else:
        main(args.Xtest, args.Ytest, model)
