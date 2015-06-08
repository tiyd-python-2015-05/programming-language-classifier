from language_classifier.lang_classifier import *

test_data = []
data_lang = 'python'

with open("language_classifier/tests/feature_test.txt") as file:
    test_file = file.read()

def test_total_characters():
    assert count_characters(test_file) == 32

def test_percent_char():
    assert percent_character(test_file, '.') == 6/32
    assert percent_character(test_file, ';') == 7/32
    assert percent_character(test_file, '\t') == 4/32

def test_count_vars():
    assert count_vars(test_file) == 2

def test_percent_word_chars():
    assert count_word_chars(test_file) == 6/32