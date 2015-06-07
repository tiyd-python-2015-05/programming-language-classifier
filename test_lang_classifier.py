from lang_classifier import *


def test_make_extension_dict():
    ext_lookup = make_extension_dict()
    assert ext_lookup['php'] == 'php'
    # assert ext_lookup['NONExISTNTANT!'] is None


def test_extract_extension():
    assert extract_extension('abc/def/ghi.jkl') == 'jkl'
    assert extract_extension('abc/def/ghi') == None


def test_load_bench_data():
    df = load_bench_data()  # reload=True)
    print(df.head(5))
    assert df['language'][2] == 'clojure'


def test_bench_data_only_contains_desired_languages():
    df = load_bench_data(reload=True)

    langs  = ['clojure', 'python', 'javascript', 'ruby', 'haskell', 'scheme',
              'java', 'scala',
              'tcl', # in reqs + tests, but no examples in bench
              'c', 'csharp', 'commonlisp', 'perl', # in reqs + bench, no tests
              'php', 'ocaml']
    training = df['language'].unique()
    for lang in langs:
        assert lang in training  # We have examples for each required language
    for lang in training:
        assert lang in langs  # We don't train for any non-required languages


def test_load_test_data():
    test_data = load_test_data()
    assert test_data['language'][1] == 'clojure'
    assert test_data['text'][2][:16] == '(ns my-cli.core)'

def setup():
    df = load_bench_data()
    X = df.text
    y = df.language
    test_data = load_test_data()
    args = train_test_split(X, y, test_size=0.2, random_state=0)
    # X_train, X_test, y_train, y_test

    return df, X, y, test_data, args

def test_assess_classifier():
    df, X, y, test_data, args = setup()
    spam_pipe = Pipeline([('bag_of_words', CountVectorizer()),
                          ('bayes', MultinomialNB())])
    classifier = assess_classifier(spam_pipe, *args)
    c = classifier.predict(X)
    assert len(c) == 585  # 923 total
    assert c[3] == 'csharp'

def test_longest_run_of_caps_feature():
    assert longest_run_of_caps_feature(
        'ABCabddwAAAA absd AB sd A.AA.AAA') == 4

def test_percent_periods_feature():
    assert percent_character_feature('.')('. . . . ') == 0.5

def test_featurizer():
    featurizer = FunctionFeaturizer(longest_run_of_caps_feature,
                                  percent_character_feature('.'))
    np.testing.assert_equal(featurizer.transform(['AAH! feature....'])
           # , np.array([[ 3.        ,  0.25]]))
           , np.array([ 3.        ,  0.25]))



"""
    df = load_bench_data()
    X = df.text
    y = df.language
    test_data = load_bench_data()

    args = train_test_split(X, y, test_size=0.2, )#random_state=0) # X_train, X_test, y_train, y_test

    spam_pipe = Pipeline([('bag_of_words', CountVectorizer()),
                          ('bayes', MultinomialNB())])
    print(spam_pipe)
    classifier = test_classifier(spam_pipe, *args)
    classifier.predict(args[1].iloc[2])

    spam_pipe = Pipeline([('bag_of_words', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                      ('RFC', RandomForestClassifier())])
    spam_pipe.set_params(RFC__n_estimators=1000)
    print(spam_pipe)
    classifier = test_classifier(spam_pipe, *args)


    test_data['guess'] = pd.DataFrame(spam_pipe.predict(test_data['text']))
    correct = test_data[test_data.language == test_data.guess]
    print('Proportion of test data correctly labeled: {:.3f}'.format(len(correct)/len(test_data)))

    longest_run_of_capitol_letters_feature('ABCabddwAAAA absd AB sd A.AA.AAA')
    percent_periods_feature('. . . . ')
    feature_vector('AAH! feature_vector... ')

    featurizer = CustomFeaturizer(longest_run_of_capitol_letters_feature,
                                  percent_periods_feature)

"""
