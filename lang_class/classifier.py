from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
import re
import numpy as np
from scipy.sparse import csc_matrix
import argparse


def main(X_test, Y_test):

    X_test = load_matrix(X_test)

    bayes = new_bayes()
    prediction = bayes.predict(X_test)


    if Y_test:
        Y_test = np.load(Y_test)

        result = sum(1 for idx in range(len(Y_test)) if Y_test[idx] != \
                  prediction[idx])

        print('success rate: ', round(100 - result / len(Y_test), 3))

    else:
        print(prediction)

def new_bayes():
    nb = MultinomialNB()
    Xtr = load_matrix('Xtr.npz')
    Ytr = np.load('Ytr.npy')
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

    parser.add_argument('--Xtest', nargs=1, default='Xte.npz', type=str)
    parser.add_argument('--Ytest', nargs=1, default='', type=str)

    args = parser.parse_args()

    main(args.Xtest, args.Ytest[0])
