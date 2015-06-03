from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import os
import argparse
import numpy as np

from file_loader import load_files


def main(name=None, thresh=.5):
    data = load_files(name)
    names = list({item for item in data[1]})
    np.save('data_keys', np.array(names))

    for idx in range(len(data[1])):
        data[1][idx] = names.index(data[1][idx])

    tf = TfidfVectorizer()
    rf = RandomForestClassifier()
    pipe = Pipeline([('tf', tf),('rf', rf)])

    cross_data = cross_validation.train_test_split(
                                       data[0], data[1], test_size=.4)

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


def save_matrix(filename, array):
    np.savez(filename, data = array.data ,indices=array.indices,
             indptr=array.indptr, shape=array.shape)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a given folder \
                                    of programs and train a classifier on \
                                    them')

    parser.add_argument('--thresh', nargs=1, default=.5, type=float)
    parser.add_argument('--name', nargs=1, default=None, type=str)

    args = parser.parse_args()

    main(args.name, args.thresh)
