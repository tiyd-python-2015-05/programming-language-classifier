import gather_data as gd
import pandas as pd
import sys
import pickle

def predict(classifier, data):
    prediction = classifier.predict(data)
    print(prediction)


if __name__ == '__main__':
    content = gd.get_snippet(sys.argv[1])
    df = pd.DataFrame(content)
    with open("./classifier", "rb") as file:
        predictor = pickle.load(file)
        predict(predictor, df)

