from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import re
import numpy as np
from scipy.sparse import csc_matrix
import argparse


def main(X_test, Y_test, data=None):
    if X_test is not None:
        X_test = load_matrix(X_test)
    else:
        if data is not None:
            X_test = data
        else:
            raise TypeError('Bad input data')

    bayes = new_bayes()
    prediction = bayes.predict(X_test)
    proba = bayes.predict_proba(X_test)

    if Y_test:
        Y_test = np.load(Y_test)

        result = sum(1 for idx in range(len(Y_test)) if Y_test[idx] != \
                  prediction[idx])

        print('success rate: ', round(100 - result / len(Y_test), 3))

    else:
        keys = np.load('data_keys.npy')
        print(keys[prediction])
        if data is not None:
            for idx in range(len(proba[0])):
                print(keys[idx], round(proba[0][idx], 3))


def new_bayes():

    Xtr = load_matrix('matrix/Xtr.npz')
    Ytr = np.load('matrix/Ytr.npy')

    nb = MultinomialNB()
    nb.fit(Xtr, Ytr)

    return nb


def load_matrix(filename):
    loader = np.load(filename)
    return csc_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a classifier using \
                                      preprocessed files, classify \
                                      input file using the classifier')

    parser.add_argument('--Xtest', nargs=1, default='matrix/Xte.npz', type=str)
    parser.add_argument('--Ytest', nargs=1, default=None, type=str)
    parser.add_argument('--from_text', nargs=1, default=None, type=str)

    args = parser.parse_args()

    if args.Ytest:
        Ytest = args.Ytest[0]
    else:
        Ytest = args.Ytest

    if args.from_text:
        pipe = joblib.load('dumps/pipe.pkl')
        from_text = args.from_text[0]
        with open(from_text, 'r') as fh:
            data = fh.read()
        data = pipe.transform([data])

        main(None, Ytest, data)

    else:
        main(args.Xtest, Ytest)
