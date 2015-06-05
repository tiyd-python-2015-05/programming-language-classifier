from glob import glob
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
# estimators
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
# other utilities
import csv
import re


def acceptable_file(text):
    if text in llist:
        return True
    else:
        return False

def clean_ext(text):
    if text == "gcc" or text == "h" or text == "gpp":
        return "c"
    elif text == "hack":
        return "php"
    elif text == "yarv" or text == "jruby":
        return "ruby"
    elif text == "clojure":
        return "clj"
    elif text == "python3" and text == "python":
        return "py"
    elif text == "perl":
        return "pl"
    elif text == "javascript":
        return "js"
    elif text == "csharp":
        return "cs"
    elif text == "ghc":
        return "hs"
    elif text == "scheme":
        return "racket"
    else:
        return text

llist = ["c", "cs", "sbcl", "clj", "hs", "java", "js",
         "ocaml", "pl", "php", "py", "ruby", "scala", "racket"]

def load_file_names():
    l = [0 for i in range(5)]
    s = "benchmarksgame-2014-08-31/benchmarksgame/"
    max_lvl = 5
    for i in range(max_lvl):
        l[i] = glob(s+"*/"*i+"*.*")
#    l[0] = glob("benchmarksgame-2014-08-31/benchmarksgame/*/*/*/*/*.*")
#    l2 = glob("benchmarksgame-2014-08-31/benchmarksgame/bench/*/*/*.*")
#    filelist = l1 + l2
    filelist = []
    for i in range(max_lvl):
        filelist += l[i]
    testlist = glob("test/*")

    print("   total samples "+str(len(filelist)))
    return filelist, testlist


def load_files(filelist, testlist):
    contents = []
    ltype = []
    ext_list = []
    for filename in filelist:
        i = filename.rfind(".")
        ext = clean_ext(filename[i+1:])
    #    print(ext, end=" - ")
    #    print(ext+ str(ext in ext_list) + " - "+str(ext_list))
        if not ext in ext_list:
            ext_list.append(ext)
        if acceptable_file(ext):
            ltype.append(ext)
            with open(filename, encoding="ISO-8859-1") as file:
    #            print(filename)
                contents.append(file.read())
#    return contents, ltype

    print(" number of usable files "+str(len(ltype)))
    print(" summary of tile types")
    for ext in ext_list:
        print(ext.ljust(12)+ "  ", end=" ")
        if ext in llist:
            print(ltype.count(ext), end=" ")
        print(" ")
    print(" not included: ", end="")
    for ext in llist:
        if ext not in ext_list:
            print(ext, end=" : ")
    print(" ")

    testcont = []
    for filename in testlist:
    #    print(filename)
        with open(filename) as file:
            testcont.append(file.read())

    print(" ")
    return contents, ltype, testcont
    #print(testcont[15])
    #print(testlist)

def read_answers():
    with open("test.csv") as csvfile:
        ans_list = csv.reader(csvfile, delimiter=",")
        ans = []
        print(ans_list)
        for row in ans_list:
            ans.append(clean_ext(row[1]))
    return ans


def fit1(contents, ltype):
    pipe = Pipeline([('bag_of_words', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('bayes', MultinomialNB())])
    pipe.fit(contents, ltype)
    return pipe
#    print(pipe.score(contents, ltype))
#    print(pipe.predict(testcont))
#    return pipe.score(contents, ltype)


def fit2(contents, ltype):
    pipe = Pipeline([('bag_of_words', CountVectorizer()),
#                          ('tfidf', TfidfTransformer()),
                          ('bayes', MultinomialNB())])
    pipe.fit(contents, ltype)
    return pipe
#    print(pipe.score(contents, ltype))
#    print(pipe.predict(testcont))
#    return pipe.score(contents, ltype)

def print_matrix(matrix, p_max=None):
    if p_max is None:
        upper_limit = len(matrix)
    else:
        upper_limit = p_max
    for i in range(upper_limit):
        vector = matrix[i]
        for val in vector:
            print(str(round(val, 3)).ljust(5)+",", end="")
        print("")
        #print([str(round(val, 3)) for val in vector])


class CustomFeaturizer:
    def __init__(self):
        pass
        #self.featurizers = featurizers

    def fit(self, X, y=None):
        """All scikit-lear compatible transforms and classifiers have the
        same interface, and fit always returns the same object."""
        return self

    def transform(self, X):
        char_list = ["^#", "\-\>", "\{", "\$", "\<", "\[", "func\b",
                    "this\.", "^end", ";", "\*", "%", "^do",
                    "\<\$php", "/\*", "__", "=", "==",
                    "===", "\(\)", "\{\}", ":", "\+\+", "\+=",
                    "^#include", "^ \*", ":\s*$", "\<\<|\>\>",
                    "int", "\b\*\w", "\(&\w", "argv", "\[\]"
                    "if\s", "if\(", "^\{", "^\}", ",\s*int\s\w",
                    "\};", "\[\d*:\d*\]", "\]\s*\{", "^//", "\w\.\{",
                    "\(\w+:", "@", "\b@\w"]
        word_list = ["private", "static", "make","let", "def", "^\(defn",
                     "defn", "do", "class", "^function", "public",
                     "unset", "printf\(", "return", "NULL", "void",
                     "main\(", "main_", "void\s\*\w", "\{else\}",
                     "char", "array\(", "__init__", "__str__", "token",
                     "^import", "^from", "final", "val", "type", "package",
                     "object", "String", "string", "primitive", "fixnum",
                     "error", "try"]
        reg_list = char_list + word_list
        matrix = []
        for text in X:
            vector = []
            for reg_expr in reg_list:
                prog = re.compile(reg_expr)
                val = len(prog.findall(text))/len(text)
                if val > 0:
                    val = 1
                vector.append(val)
            matrix.append(vector)
        return matrix


def fit3(contents, ltype):
    custom_feature = CustomFeaturizer()
    pipe = make_pipeline(custom_feature, DecisionTreeClassifier())
    pipe.fit(contents, ltype)
    return pipe


def fit4(contents, ltype):
    custom_feature = CustomFeaturizer()
    pipe = make_pipeline(custom_feature, SGDClassifier())
    pipe.fit(contents, ltype)
    return pipe


#sms_featurizer = CustomFeaturizer(longest_run_of_capital_letters_feature,
#                                  percent_periods_feature)
#big_list = sms_featurizer.transform(sms_data[:10])
#print(big_list)

if __name__ == "__main__":
    filelist, testlist = load_file_names()
    contents, ltype, testcont = load_files(filelist, testlist)

    plist = [fit1, fit2, fit3, fit4]

    X, Xt, y, yt = train_test_split(contents, ltype, test_size=0.33)
    pipel = [0 for i in range(len(plist))]
    for i in range(len(plist)):
        pipel[i] = plist[i](X, y)
    #pipe1 = fit1(contents, ltype)
    #pipe2 = fit2(contents, ltype)

    ans = read_answers()
    print(ans)

    i = 0
    for pipe in pipel:
        i += 1
        print(" score_train "+str(i)+" "+str(pipe.score(X, y)))
        print(" score_test  "+str(i)+" "+str(pipe.score(Xt, yt)))
        print(" score_quest "+str(i)+" "+str(pipe.score(testlist, ans)))
        print(" pred "+str(i)+" "+str(pipe.predict(testlist)))
        print(" ")

    word_list = re.findall(r"^#", "# include ")
    print(word_list)
    print(len(word_list))
