from bs4 import BeautifulSoup
import requests
import urllib
from re import findall
import pandas as pd
import random
import pickle

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


def scrape(num_links=30, drop_less_than=0, save=False):
    df = make_data(make_links_list(num_links))
    ndf = df[df[0] != 'text']
    ndf = ndf.groupby(0).filter( lambda x: len(x) >= drop_less_than)
    if save:
        ndf.to_pickle('data/{}.pkl'.format(num_links))
    return ndf


def scrape_filter(num_links=30, drop_less_than=0, save=False):
    df = make_data(make_links_list(num_links))
    df = df[df[0] != 'text']
    df = df[(df[0] == 'ada') | (df[0] == 'clojure') | (df[0] == "algol68") | (df[0] == "awk")
            | (df[0] == "bash") | (df[0] == "haskell") | (df[0] == "java") | (df[0] == "javascript") | (df[0] == "lisp") | (df[0] == "objc") | (df[0] == "ocaml") | (df[0] == "php") | (df[0] == "python") | (df[0] == "ruby") | (df[0] == "scala") | (df[0] == "scheme") | (df[0] == "tcl")]
    df = df.groupby(0).filter(lambda x: len(x) >= drop_less_than)
    if save:
        name = "data/filtered_{}_of_{}.pkl".format(drop_less_than, num_links)
        df.to_pickle(name)
    return df


def scraper_filter_small(num_links=30, drop_less_than=0, save=False):
    df = make_data(make_links_list(num_links))
    df = df[df[0] != 'text']
    df = df[(df[0] == 'clojure') | (df[0] == "haskell") | (df[0] == "java") | (df[0] == "javascript") | (df[0] == "ocaml") | (df[0] == "php") | (df[0] == "python") | (df[0] == "ruby") | (df[0] == "scala") | (df[0] == "scheme") | (df[0] == "tcl")]
    df  = df.groupby(0).filter(lambda x: len(x) >= drop_less_than)
    if save:
        name = "data/smaller_{}_of_{}.pkl".format(drop_less_than, num_links)
        df.to_pickle(name)
    return df
    
def load_data(file_name):
    return pd.read_pickle('data/{}.pkl'.format(file_name))
