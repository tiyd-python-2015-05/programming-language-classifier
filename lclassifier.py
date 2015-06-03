from glob import glob

l1 = glob("benchmarksgame-2014-08-31/benchmarksgame/bench/binarytrees/*")
filelist = glob("benchmarksgame-2014-08-31/benchmarksgame/bench/*/*.*")

print(str(len(l1))+" l2 "+str(len(filelist)))

contents = []
ltype = []
for filename in filelist:
    if "ocaml-2" not in filename:
        i = filename.index(".")
        ltype.append(filename[i:])
        with open(filename) as file:
            contents.append(file.read())

testcont = []
testlist = glob("test/*")
for filename in testlist:
    print(filename)
    with open(filename) as file:
        testcont.append(file.read())

print(" ")
print(ltype)
print(" ")
#print(testcont[15])
#print(testlist)

#from scikit-learn.datasets import load_iris
from sklearn import datasets
iris = datasets.load_iris()
print(iris.keys())
print(" ")
#print(iris.data)
print(" ")
print(iris.target)

from sklearn import neighbors, datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

# create the model
knn = neighbors.KNeighborsClassifier(n_neighbors=5)

# fit the model
knn.fit(X, y)

# What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
# call the "predict" method:
result = knn.predict([[3, 5, 4, 2],])

print(iris.target_names[result])



import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer



pipe = Pipeline([('bag_of_words', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('bayes', MultinomialNB())])

pipe.fit(contents, ltype)

print(pipe.score(contents, ltype))

print(pipe.predict(testcont))
