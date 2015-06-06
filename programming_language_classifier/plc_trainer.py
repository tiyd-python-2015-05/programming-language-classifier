import re
import itertools
import get_data as gd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle


def percent_elements(text):
    elements = ")}];:.,\/-_#*!$%|<>& "
    results = []
    for element in elements:
        total = max(1, len(text))
        results.append(text.count(element)/total)
    return results


def number_elements(text):
    elements = [r'\bbegin\b', r'\bend\b', r'\bdo\b', r'\bvar\b', r'\bdefine\b', r'\bdefn\b', r'\bfunction\b',
                r'\bclass\b', r'\bmy\b', r'\brequire\b', r'\bvoid\b', r'\bval\b', r'\bpublic\b', r'\blet\b',
                r'\bwhere\b', r'\busing\b', r'\bextend\b', r'\bfunction\b']
    results = []
    for element in elements:
        results.append(len(re.findall(element, text)))
    return results


def longest_run(text):
    elements = [r'[)]+',r'[}]+', r'[\]]+', r'[=]+']
    results = []
    for element in elements:
        runs = sorted(re.findall(element, text), key=len)
        if runs:
            results.append(len(runs[-1]))
        else:
            results.append(0)
    return results


def line_enders(text):
    elements = [r'[)]$', r';$', r'}$', r']$', r'\):$']
    results = []
    for element in elements:
        results.append(len(re.findall(element, text, re.MULTILINE)))
    return results


class Featurizer:
    def __init__(self, *feature_makers):
        self.feature_makers = feature_makers

    def fit(self, X, y):
        return self

    def transform(self, X):
        feature_vectors = []
        for item in X:
            vector = list(itertools.chain.from_iterable([function(item) for function in self.feature_makers]))
            feature_vectors.append(vector)
        return feature_vectors



if __name__ == '__main__':
    content_list = gd.get_content("./train/")
    train_data = gd.make_dataframe(content_list)
    classifier = Pipeline([('features', Featurizer(percent_elements, number_elements, longest_run, line_enders)),
                           ('bayes', MultinomialNB())])
    classifier.fit(train_data[1], train_data[0])
    with open("./classifier", "wb") as file:
        pickle.dump(classifier, file)
