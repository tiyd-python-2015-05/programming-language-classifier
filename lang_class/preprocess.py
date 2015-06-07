#!/usr/bin/env python3

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import os
import argparse
import numpy as np

from file_loader import load_files
from vectorizer import CodeVectorizer


def main(name=None, thresh=.3, vectorizer=None):
    """
    breaks the data into test/train cases
    and trains a vectorizer and random forest
    on the test data, then saves the results
    to Xtr.npz, Ytr.npy, Xte.npz, Yte.npy
    and pickles the pipeline
    """
    data = load_files(name)
    pipe = make_pipe(vectorizer)

    cross_data = cross_validation.train_test_split(
                                       data[0], data[1], test_size=thresh)

    pipe.fit(cross_data[0], cross_data[2])

    cross_data[0] = pipe.transform(cross_data[0])
    cross_data[1] = pipe.transform(cross_data[1])

    cross_data[2] = np.array(cross_data[2])
    cross_data[3] = np.array(cross_data[3])

    save_matrix('matrix/Xtr', cross_data[0])
    np.save('matrix/Ytr', cross_data[2])

    save_matrix('matrix/Xte', cross_data[1])
    np.save('matrix/Yte', cross_data[3])

    joblib.dump(pipe, 'dumps/pipe.pkl')


def make_pipe(vectorizer=None):
    '''
    creates a pipeline with the given
    vectorizer or the Tfidf if none given
    and a random forest classifier
    '''
    if vectorizer is None:
        tf = TfidfVectorizer(sublinear_tf=True, token_pattern= \
                r'\b[\w\.,:\(\)\[\]\{\}\'\";%#@!*&|\<\>]+\b', stop_words=None,
                binary=True)
    else:
        tf = vectorizer

    rf = RandomForestClassifier()
    return Pipeline([('tf', tf), ('rf', rf)])

def save_matrix(filename, array):
    """
    saves the given matrix to a .npz file
    """
    np.savez(filename, data = array.data ,indices=array.indices,
             indptr=array.indptr, shape=array.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a given folder \
                                    of programs and train a classifier on \
                                    them')

    parser.add_argument('--thresh', default=.5, type=float, help='specify \
                        a test/train split %')

    parser.add_argument('--name', default=None, type=str, help='specify \
                        a source for the test data')

    parser.add_argument('-cv', help= 'Use CodeVectorizer rather than \
                        TfidfVectorizer with altered tokenizer',
                        action='store_true')

    vect=None
    args = parser.parse_args()

    if args.cv:
        vect = CodeVectorizer(TfidfVectorizer(binary=True, norm='l1'))

    main(args.name, args.thresh, vect)
