from lclassifier import *

def test_ext():
    ext = "cowboy"
    assert acceptable_file(ext) == False

def test_correct_ext():
    ext = "perl"
    assert clean_ext(ext) == "pl"
