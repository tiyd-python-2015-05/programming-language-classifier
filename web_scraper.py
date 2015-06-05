from bs4 import BeautifulSoup
import requests
import urllib
from re import findall
import pandas as pd
import random
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import re



languages_list = ['ACL2',
 'Ada',
 'Aime',
 'ALGOL 68',
 'AppleScript',
 'AutoHotkey',
 'AutoIt',
 'AWK',
 'BASIC',
 'BBC BASIC',
 'bc',
 'Brat',
 'C',
 'C++',
 'C#',
 'Clojure',
 'COBOL',
 'CMake',
 'CoffeeScript',
 'Common Lisp',
 'D',
 'Delphi',
 'DWScript',
 'E',
 'Eiffel',
 'Erlang',
 'ERRE',
 'Euphoria',
 'Factor',
 'Fantom',
 'Forth',
 'Fortran',
 'Frink',
 'F#',
 'FunL',
 'GAP',
 'Go',
 'Groovy',
 'Haskell',
 'Icon and Unicon',
 'Inform 6',
 'J',
 'Java',
 'JavaScript',
 'Joy',
 'Julia',
 'LabVIEW',
 'Lasso',
 'Liberty BASIC',
 'Logo',
 'Lua',
 'M4',
 'Mathematica',
 'MATLAB',
 'Maxima',
 'Modula-3',
 'MUMPS',
 'Nemerle',
 'NetRexx',
 'Nim',
 'Objective-C',
 'OCaml',
 'Oforth',
 'Oz',
 'PARI/GP',
 'Pascal',
 'Perl',
 'Perl 6',
 'PHP',
 'PicoLisp',
 'PL/I',
 'PowerShell',
 'PureBasic',
 'Python',
 'R',
 'Racket',
 'REBOL',
 'REXX',
 'Ruby',
 'Run BASIC',
 'Rust',
 'Scala',
 'Scratch',
 'Seed7',
 'Sidef',
 'Smalltalk',
 'SNOBOL4',
 'Swift',
 'Tcl',
 'TI-83 BASIC',
 'TUSCRIPT',
 'UNIX Shell',
 'Ursala',
 'VBScript',
 'Vedit macro language',
 'zkl']

def get_text(url):
    """Takes a url and returns text"""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    content = urllib.request.urlopen(req).read()
    page_text=BeautifulSoup(content)
    return page_text.get_text()

# def scrape_text(text):
#     data_crop = findall("[EDIT] \n.+\n", text)
#     return data_crop


# def scrape_text(text):
#     """Takes text from get_text and returns a list of tuples with
#     language in [0] and code in [1]"""
#     data_crop = findall(r"edit] (.+)\n(.+)\n", text)
#     return data_crop
#     ##Should maybe grab all of the text

# def scrape_links():
#     """Creates list of links to use with create_url to gather code."""
#     with open ("links_list.txt", "r") as myfile:
#         data=myfile.read()
#     return findall(r"wiki/(.+)\" ti", data)
#
# language_start = ["C", "C#", "Common Lisp", "Clojure", "Haskell",
#                   "Java", "JavaScript", "OCaml", "Perl", "PHP",
#                   "Python", "Ruby", "Scala", "Scheme"]


#def make_data(languages=language_start, num_links=50)
    #grab data for all of the links in the task list
    #go through for each of the languages and grab the associated
    #code
    #return a df with the code you need in a column and the type of
    #code as the index


def scrape_data(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    content = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(content)
    return soup.find_all( "pre", class_="highlighted_source")
    #pre is an html tag. We want all text from pre with class highlighted_source
    #returns a list of soup objects


def pull_code_from_soup(soup_list):
    return [[soup_list[i]['class'][0], soup_list[i].get_text()] for i in range(len(soup_list))]


def make_data(url_list):
    code_snippets = pd.DataFrame(columns=([0, 1]))
    for url in url_list:
        soup_list = scrape_data(url)
        url_df = pd.DataFrame(pull_code_from_soup(soup_list))
        code_snippets = code_snippets.append(url_df, ignore_index=True)
    return code_snippets


def scrape_links():
    req = urllib.request.Request('http://rosettacode.org/wiki/Category:Programming_Tasks', headers={'User-Agent': 'Mozilla/5.0'})
    content = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(content)
    link_list = [link.get('href') for link in soup.find_all('a')]
    return ["http://www.rosettacode.org{}".format(link) for link in link_list[1:] if link.startswith('/wiki/')]


def make_links_list(num_links=30):
    return random.sample(scrape_links(), num_links)


def scrape(num_links=30):
    df = make_data(make_links_list(num_links))
    return df[df[0] != 'text']


def scraper(num_links=50, min_examples=25):
    df = make_data(make_links_list(num_links))
    df = df[df[0] != 'text']
    return df.groupby(0).filter(lambda x: len(x) >= min_examples)


def split_fit_score(dataframe, estimator="Bayes"):
    df_X = dataframe.loc[:, 1]
    df_y = dataframe.loc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y)
    if estimator == "Bayes":
        new_pipe = Pipeline([("bag_of_words", CountVectorizer()),
                   ("nb", MultinomialNB())])
    elif estimator == "Gaussian":
        new_pipe = Pipeline([("bag_of_words", CountVectorizer()),
                   ("gnb", GaussianNB())])
    elif estimator == "Bernoulli":
        new_pipe = Pipeline([("bag_of_words", CountVectorizer(binary=True)),
                   ("bnb", BernoulliNB())])
    new_pipe.fit(X_train, y_train)
    return new_pipe.score(X_test, y_test)