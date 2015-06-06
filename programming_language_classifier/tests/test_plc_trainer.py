import programming_language_classifier.plc_trainer as plc


def test_percent_elements():
    """element order: ) } ] ; : . , \ / - _ # * ! $ % | """
    a_string = "..oooooOO}"
    assert plc.percent_elements(a_string) == [0, 0.1, 0, 0, 0, .2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    a_string = "]]]*%!,,:M"
    assert plc.percent_elements(a_string) == [0, 0, 0.3, 0, 0.1, 0, 0.2, 0, 0, 0, 0, 0, 0.1, 0.1, 0, 0.1, 0, 0, 0, 0, 0]
    a_string = ""
    assert plc.percent_elements(a_string) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_number_elements():
    """element order: begin end do"""
    a_string = "begin: words!!! end begin itbeginq"
    assert plc.number_elements(a_string) == [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    a_string = "dobeginend do do end, Mend :begin:"
    assert plc.number_elements(a_string) == [1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    a_string = ""
    assert plc.number_elements(a_string) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_longest_run():
    """element order: ) } ] ="""
    a_string = ")))))[]]]}]]]]}}="
    assert plc.longest_run(a_string) == [5, 2, 4, 1]
    a_string = "Adn;ksenfas]]]]]((()===="
    assert plc.longest_run(a_string) == [1, 0, 5, 4]


def test_line_enders():
    a_string = "....)\n ....;\n....;\n"
    assert plc.line_enders(a_string) == [1, 2, 0, 0, 0]


def test_featurizer_transform():
    tf = plc.Featurizer(plc.percent_elements, plc.number_elements, plc.longest_run)
    test_list = ["begin }}} . end", "do end %%__=====", ""]
    array = tf.transform(test_list)
    test_array = [[0, 0.2, 0, 0, 0, 0.06666667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2,
                   1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 3, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.125, 0, 0, 0, 0, 0.125, 0, 0, 0, 0, 0.125,
                   0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 5],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0]]
    def rounder(array):
        for collection in array:
            for index in range(len(collection)):
                collection[index] = round(collection[index], 3)
        return array

    assert rounder(array) == rounder(test_array)