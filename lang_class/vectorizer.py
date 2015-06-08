import re
import numpy as np
from scipy.sparse import csc_matrix, hstack, coo_matrix



class CodeVectorizer():
    def __init__(self, vectorizer):
        self.fit_X = None
        self.fit_Y = None
        self.punctuation = None
        self.lengths = None
        self.fitted = False
        self.punc = '. , ; : ! # $ % * ? + - & ^ | = _'.split(' ')
        self.vectorizer = vectorizer

    def fit(self, X, y):
        self.fit_X = X
        self.fit_Y = y
        self.process()
        self.vectorizer.fit(X, y)
        self.fitted = True

    def find_brackets(self, X): # assumes all delimiters are matched
        if isinstance(X, list):
            final = []
            for item in X:
                final.append(self.find_brackets(item))
            return final

        delimiters = re.finditer(r'([\{\(\[\]\)\}])', X)
        positions = [(item.groups(0)[0], item.span()[0]) for item in delimiters]
        left = ['(', '[', '{']

        if len(positions)%2 != 0:
            return 0

        if len([item[0] for item in positions if item in left]) != \
           len(positions)/2:
            return 0

        final = [len(positions)]
        idx = 0
        while positions:

            if positions[idx][0] not in left:
                final.append(positions[idx][1] - positions[idx - 1][1])
                positions.pop(idx)
                positions.pop(idx-1)

                if idx > 1:
                    idx -= 2

            idx += 1

            if idx >= len(positions):
                idx = 0

        return final

    def find_punctuation(self, X):
        if isinstance(X, list):
            return [[line.count(item) for item in self.punc] for line in X]
        return [x.count(item) for item in self.punc]

    def process(self):
        self.lengths = self.find_brackets(self.fit_X)
        self.punctuation = self.find_punctuation(self.fit_X)

    def transform(self, X):
        if self.fitted:
            ln = coo_matrix(np.matrix(self.find_brackets(X))).transpose()
            pc = coo_matrix(np.matrix(self.find_punctuation(X)))
            X_transformed = self.vectorizer.transform(X)

            if not isinstance(X_transformed, csc_matrix):
                X_transformed = csc_matrix(X_transformed)

            return hstack([X_transformed, ln, pc], format='csc')
        else:
            raise Exception('Did not fit before transforming')

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
