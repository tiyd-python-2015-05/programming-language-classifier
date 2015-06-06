from bs4 import BeautifulSoup
import urllib
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def get_text(url):
    """Takes a url and returns text"""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    content = urllib.request.urlopen(req).read()
    page_text=BeautifulSoup(content)
    return page_text.get_text()


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


def scraper(num_links=50, min_examples=25, save=False):
    df = make_data(make_links_list(num_links))
    df = df[df[0] != 'text']
    df = df.groupby(0).filter(lambda x: len(x) >= min_examples)
    if save:
        name = "scraper_{}_x_{}.pkl".format(num_links, min_examples)
        df.to_pickle(name)
    return df


def scraper_filter(num_links=50, min_examples=1, save=False):
    df = make_data(make_links_list(num_links))
    df = df[df[0] != 'text']
    df = df[(df[0] == 'ada') | (df[0] == 'clojure') | (df[0] == "algol68") | (df[0] == "awk")
                     | (df[0] == "bash") | (df[0] == "haskell") | (df[0] == "java") | (df[0] == "javascript")
                     | (df[0] == "lisp") | (df[0] == "objc") | (df[0] == "ocaml") | (df[0] == "php")
                     | (df[0] == "python") | (df[0] == "ruby") | (df[0] == "scala") | (df[0] == "scheme")
                     | (df[0] == "tcl")]
    df = df.groupby(0).filter(lambda x: len(x) >= min_examples)
    if save:
        name = "scraper_filter_{}_x_{}.pkl".format(num_links, min_examples)
        df.to_pickle(name)
    return df


def scraper_filter_small(num_links=50, min_examples=1, save=False):
    df = make_data(make_links_list(num_links))
    df = df[df[0] != 'text']
    df = df[(df[0] == 'clojure') | (df[0] == "haskell") | (df[0] == "java") | (df[0] == "javascript")
        | (df[0] == "ocaml") | (df[0] == "php") | (df[0] == "python") | (df[0] == "ruby")
        | (df[0] == "scala") | (df[0] == "scheme") | (df[0] == "tcl")]
    df = df.groupby(0).filter(lambda x: len(x) >= min_examples)
    if save:
        name = "scraper_filter_{}_x_{}.pkl".format(num_links, min_examples)
        df.to_pickle(name)
    return df





# Should we make this a class?
def split_fit_score(dataframe, estimator="Bayes", report=False):
    df_X = dataframe.loc[:, 1]
    df_y = dataframe.loc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y)
    if estimator == "Bayes":
        new_pipe = Pipeline([("bag_of_words", CountVectorizer()),
                   ("nb", MultinomialNB())])
    elif estimator == "Forest":
        new_pipe = Pipeline([("bag_of_words", CountVectorizer()),
                   ("forest", RandomForestClassifier())])
    elif estimator == "neighbors":
        new_pipe = Pipeline([("bag_of_words", CountVectorizer(binary=True)),
                   ("neighbors", KNeighborsClassifier())])
    new_pipe.fit(X_train, y_train)
    if report:
        return (classification_report(new_pipe.predict(X_test), y_test))
    else:
        return new_pipe.score(X_test, y_test)


