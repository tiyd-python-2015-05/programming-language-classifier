import pandas as pd
import glob
from bs4 import BeautifulSoup
import os

extension_dict = {'gcc': 'c', 'perl': 'perl', 'clojure': 'clojure', 'hs': 'haskell', 'java': 'java',
                 'javascript': 'javascript', 'jruby': 'ruby', 'yarv': 'ruby', 'ocaml': 'ocaml',
                 'sbcl': 'lisp', 'scala': 'scala', 'csharp': 'csharp', 'hack': 'php', 'php': 'php',
                 'python3': 'python', 'racket': 'scheme', 'tcl': 'tcl'}

def get_test_data():
    content = []
    for file in sorted(os.listdir("../data/test/"), key=int):
        with open("../data/test/" + file) as fh:
            content.append([fh.read()])
    test_data = pd.DataFrame(content)
    return test_data

def get_code_from_html(lang):
    htmlfiles = glob.glob("../data/html/*.html")
    texts = []
    tags = []
    for file in htmlfiles:
        soup = BeautifulSoup(open(file))
        html_tag = soup.find_all('pre', {'class' : '{} highlighted_source'.format(lang)})
        html_text = [part.get_text() for part in html_tag]
        for tag in html_tag:
            tags.append(lang)
        texts.extend(html_text)
    return texts, tags

def get_benchmark_code(directory):
    files = glob.glob("../data/corpus/{}/*.{}".format(directory, directory))
    texts = []
    tags = []
    for file in files:
        with open(file) as fh:
            tags.append(extension_dict[directory])
            texts.append(fh.read())
    return texts, tags

def get_snippet(filename):
    content = []
    with open(filename) as fh:
        content.append([fh.read()])
    return content