from bs4 import BeautifulSoup
import urllib
from re import findall
import pandas as pd
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score
import pickle

# language_start = ["C", "C#", "Common Lisp", "Clojure", "Haskell",
#                   "Java", "JavaScript", "OCaml", "Perl", "PHP",
#                   "Python", "Ruby", "Scala", "Scheme"]
#

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
        code_snippets = code_snippets.append(pd.DataFrame(pull_code_from_soup(soup_list)), ignore_index=True)
    return code_snippets


def scrape_links():
   req = urllib.request.Request('http://rosettacode.org/wiki/Category:Programming_Tasks', headers={'User-Agent': 'Mozilla/5.0'})
   content = urllib.request.urlopen(req).read()
   soup = BeautifulSoup(content)
   link_list = [link.get('href') for link in soup.find_all('a')]
   return ["http://www.rosettacode.org{}".format(link) for link in link_list[1:] if link.startswith('/wiki/')]


def make_links_list(num_links=30):
    return random.sample(scrape_links(), num_links)


def scrape_and_clean(num_links=30):
    df = make_data(make_links_list(num_links))
    new_df = df[df[0]!='text']
    return new_df

def scrape_clean_cut(num_links=100, min_examples=40, save=False):
    df = make_data(make_links_list(num_links))
    new_df = df[df[0]!='text']
    new_df = new_df.groupby(0).filter(lambda x: len(x) >= min_examples)
    if save:
        new_df.to_pickle("scraper_{}x{}.pkl".format(num_links, min_examples))
    return new_df


def scrape_clean_cut_filter(num_links=100, min_examples=40, save=False):
    df = make_data(make_links_list(num_links))
    df = df[df[0]!='text']
    new_df = df[(df[0] == 'clojure') | (df[0] == 'haskell') | (df[0] == 'java') | (df[0] == 'javascript')
                | (df[0] == 'ocaml') | (df[0] == 'php') | (df[0] == 'python') | (df[0] == 'ruby')
                | (df[0] == 'scala') | (df[0] == 'scheme') | (df[0] == 'tcl')]
    
    new_df = new_df.groupby(0).filter(lambda x: len(x) >= min_examples)
    if save:
        new_df.to_pickle("scraper_filtered_{}x{}.pkl".format(num_links, min_examples))
    return new_df

def pipeline_runner(dataframe, estimator='Multinomial', report=False):
    """Should have made this a class not a function"""
    ##Re-testing with MultinomialNB
    y = dataframe.loc[:, 0]
    X = dataframe.loc[:, 1]
    #splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    #running pipe to vectorize and run estimator
    if estimator == 'Multinomial':
        estimator_pipe = Pipeline([('bag_of_words', CountVectorizer()),
                              ('mnb', MultinomialNB())])
    elif estimator == 'KNeighbors':
        estimator_pipe = Pipeline([('bag_of_words', CountVectorizer()),
                              ('knn', KNeighborsClassifier())])
    elif estimator == 'Forest':
        estimator_pipe = Pipeline([('bag_of_words', CountVectorizer()),
                              ('forest', RandomForestClassifier())])
    else:
        return pipeline_runner(dataframe, estimator)
    #fitting
    estimator_pipe.fit(X_train, y_train)
    #checking score
    if report:
        return (classification_report(estimator_pipe.predict(X_test), y_test))
    else:
        return estimator_pipe.score(X_train, y_train), estimator_pipe.score(X_test, y_test)


