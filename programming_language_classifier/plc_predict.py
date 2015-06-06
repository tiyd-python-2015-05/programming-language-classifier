import os
import sys
import pickle
import get_data as gd
from plc_trainer import Featurizer, percent_elements, number_elements, longest_run, line_enders

def predict(classifier, directory):
    content = []
    for filename in os.listdir(directory):
        with open(directory + filename) as fh:
            content.append([filename, fh.read()])
    test_data = gd.make_dataframe(content)
    predictions = list(classifier.predict(test_data[1]))
    buffer = max([len(item) for item in test_data[0]]) + 5
    for index in range(len(predictions)):
        print(test_data[0][index].ljust(buffer) + "| " + predictions[index])



if __name__ == '__main__':
    with open("./classifier", "rb") as file:
        predictor = pickle.load(file)
        predict(predictor, sys.argv[1])
