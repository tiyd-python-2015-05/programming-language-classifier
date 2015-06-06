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
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import TransformerMixin
# other utilities
import csv
import re


def acceptable_file(text):
    if text in llist:
        return True
    else:
        return False

def clean_ext(textp):
    text = textp.strip()
    if text == "gcc" or text == "h" or text == "gpp":
        return "c"
    elif text == "hack":
        return "php"
    elif text == "yarv" or text == "jruby":
        return "ruby"
    elif text == "clojure":
        return "clj"
    elif text == "python3" or text == "python":
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

def list_uniques(alist):
    rlist = []
    for item in alist:
        if item not in rlist:
            rlist.append(item)
    return rlist

def load_file_names():
    l = [0 for i in range(5)]
    s = "benchmarksgame-2014-08-31/benchmarksgame/bench/"
    max_lvl = 4
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
        if ext == "tcl":
            print(filename)
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
    print(" ")
    print(" number of read file types:  "+str(len(ext_list)))
    print(" number of recognized types: "+str(len(llist)))
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

    testcont = [0] * 32
    for filename in testlist:
    #    print(filename)
        with open(filename) as file:
            di = filename.rfind("/")
            i = int(filename[di+1:])
#            print(filename+" "+str(i))
            testcont[i-1] = file.read()
    print(" ")
    return contents, ltype, testcont
    #print(testlist)

def read_answers():
    with open("test.csv") as csvfile:
        ans_list = csv.reader(csvfile, delimiter=",")
        ans = []
        print(ans_list)
        for row in ans_list:
            ans.append(clean_ext(row[1]))
    print(" number of testing file types: "+str(len(list_uniques(ans))))
#            print(row[0])
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


class CustomFeaturizer(TransformerMixin):
    def __init__(self):
        pass
        #self.featurizers = featurizers

    def fit(self, X, y=None):
        """All scikit-lear compatible transforms and classifiers have the
        same interface, and fit always returns the same object."""
        return self

    def transform(self, X):
        # char_list = ["^#", "\-\>", "\{", "\$", "\<", "\[", "func\b",
        #             "this\.", "^end", ";", "\*", "%", "^do",
        #             "\<\$php", "/\*", "__", "=", "==",
        #             "===", "\(\)", "\{\}", ":", "\+\+", "\+=",
        #             "^#include", "^ \*", ":\s*$", "\<\<|\>\>",
        #             "int", "\b\*\w", "\(&\w", "argv", "\[\]"
        #             "if\s", "if\(", "^\{", "^\}", ",\s*int\s\w",
        #             "\};", "\[\d*:\d*\]", "\]\s*\{", "^//", "\w\.\{",
        #             "\(\w+:", "@", "\b@\w"]
        # word_list = ["private", "static", "make","let", "def", "^\(defn",
        #              "defn", "do", "class", "^function", "public",
        #              "unset", "printf\(", "return", "NULL", "void",
        #              "main\(", "main_", "void\s\*\w", "\{else\}",
        #              "char", "array\(", "__init__", "__str__", "token",
        #              "^import", "^from", "final", "val", "type", "package",
        #              "object", "String", "string", "primitive", "fixnum",
        #              "error", "try"]
        cish = ["^[ \t]*\*", "^[ \t]*/\*\*"]
        clojure = ["^\s*\(\w.*\s*\)$", "^[ \t]*;", "\(def(n)? "]
        python = ["\):[ \t]*\n[ \t]*\w", "\s__\w*__\(", "(^from|^import)\s",
                  "def\s*\w*\([ \w,]*\):[ \t]*\n(( {4})+|\t+)\w"]
        js = ["^[ \t]*var", "=\s*function",
              "function\s*\w*\(\w*[\w\s,]*\)\s*\{"]
        ruby = ["^[ \t]*end$", "^[ \t]*def *\w*(\(\w*\))?[ \t]*$",
                "^[ \t]*include \w*[ \t]*$", "^[ \t]*@", "super"]
        hs = ["&&&", "^\{-"]
        clj = ["^\(define", "^[ \t]*;+"]
        java = ["^[ \t]*public \w* \w*", "^import .*;$"]
        scl = ["^[ \t]*object \w*", "^[ \t]*(final)?val \w* ="]
        tcl = ["^[ \t]*proc \w*::\w* \{"]
        php = ["^[ \t]*(\w*)?( )?function \w*( )?\(&?\$\w*",
                "^[ \t]*\$\w* ?=.*;$"]
        ocaml = ["^[ \t]*let \w+", "^[ \t]*struct[ \t]*$"]
        perl = ["^[ \t]*my ", "^[ \t]*sub \w* \{"]
        gcc = ["^[ \t]*typedef \w* \w* ?\{", "^#include ?\<",
               "^using .*;$", "sealed"]
#        reg_list = char_list + word_list
        reg_list = clojure + python + js + ruby + hs + clj + java + scl\
                   + tcl + php + ocaml + perl + gcc + cish
#        print(len(reg_list))
        matrix = []
        for text in X:
            v = [0] * len(reg_list)
#            print(str(len(v))+" "+str(len(reg_list)))
            for i in range(len(reg_list)):
#                print(reg_expr)
                reg_expr = reg_list[i]
                prog = re.compile(reg_expr, flags=re.MULTILINE)
                val = len(prog.findall(text))#/len(text)
                #if val > 0:
                #    val = 1
#                print(i)
                v[i] = val
#            print(vector)
            matrix.append(v)
#        print(matrix[0])
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


def fit5(contents, ltype):
    custom_feature = CustomFeaturizer()
    pipe = make_pipeline(custom_feature, MultinomialNB())
    pipe.fit(contents, ltype)
    return pipe


def fit6(contents, ltype):
    custom_feature = CustomFeaturizer()
    pipe = make_pipeline(custom_feature, RandomForestClassifier())
    pipe.fit(contents, ltype)
    return pipe


def demo_class(X, y):
    types = []
    for ext in y:
        if ext not in types:
            types.append(ext)
    typecont = [""] * len(types)
    for i in range(len(X)):
        text = X[i]
        for j in range(len(types)):
            ext = types[j]
            if ext == y[i]:
                typecont[j] += text
    custom_feature = CustomFeaturizer()
    M = custom_feature.transform(typecont)
    for j in range(len(M)):
        print(types[j].ljust(8)+" ", end="")
        for k in range(len(M[0])):
            print(str(int(M[j][k])).ljust(5), end="")
        print("")

#sms_featurizer = CustomFeaturizer(longest_run_of_capital_letters_feature,
#                                  percent_periods_feature)
#big_list = sms_featurizer.transform(sms_data[:10])
#print(big_list)

if __name__ == "__main__":
    filelist, testlist = load_file_names()
    contents, ltype, testcont = load_files(filelist, testlist)

    plist = [fit2, fit3, fit4, fit5, fit6]

    X, Xt, y, yt = train_test_split(contents, ltype, test_size=0.33)
    pipel = [0 for i in range(len(plist))]
    for i in range(len(plist)):
        pipel[i] = plist[i](X, y)
    #pipe1 = fit1(contents, ltype)
    #pipe2 = fit2(contents, ltype)
    pipe = fit4(X, y)
    M = pipe.transform(testcont)
    print(str(len(M))+" "+str(len(M[0])))
#    print(M[0])
    M = pipe.transform(Xt)
    print(str(len(M))+" "+str(len(M[0])))
    print("  failed to classify")
    A = pipe.predict(X)
    for i in range(len(A)):
        if A[i] != y[i]:
#            print(" ")
            print(y[i].ljust(6)+" misclassified as "+A[i])
#            print(X[i])
#    print(M[0])


    cf = CustomFeaturizer()
    M = cf.transform(testcont)
    print(str(len(M))+" "+str(len(M[0])))
#    print(M[0])
    M = cf.transform(Xt)
    print(str(len(M))+" "+str(len(M[0])))
#    print(M[0])
    #print(testcont)

    ans = read_answers()
    print(ans)

    i = 0
    for pipe in pipel:
        i += 1
        print(" score_train "+str(i)+" "+str(pipe.score(X, y)))
        print(" score_test  "+str(i)+" "+str(pipe.score(Xt, yt)))
        print(" score_quest "+str(i)+" "+str(pipe.score(testcont, ans)))
        print(" pred "+str(i)+" "+str(pipe.predict(testcont)))
        print(" ")

    word_list = re.findall(r"^#", "# include ")
    print(word_list)
    print(len(word_list))

    demo_class(testcont, ans)
