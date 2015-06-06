from programming_language_classifier import get_data as gd


def test_get_content():
    assert gd.get_content("tests/function_testfiles/") == [["C", "This is a C file\n"],
                                                    ["JavaScript", "This is a javascript file\n"],
                                                    ["Ruby", "This is a Ruby file\n"],
                                                    ["Python", "This is a Python file\n"]]


def test_make_dataframe():
    test_list = gd.get_content("tests/function_testfiles/")
    assert gd.make_dataframe(test_list)[0][0] == "C"
    assert gd.make_dataframe(test_list)[1][0] == "This is a C file\n"
    assert gd.make_dataframe(test_list)[1][2] == "This is a Ruby file\n"


'javascript': '.js',
             'haskell': '.haskell',
             'scala': '.scala',
             'ocaml': '.ocaml',
             'ruby': '.jruby',
             'php': '.php',
             'clojure': '.clojure',
             'perl': '.perl',
             'csharp': '.csharp',
             'java': '.java',
             'c': '.gcc',
             'scheme': '.racket',
             'python': '.py',
             'lisp': '.sbcl',

