from lclassifier import *

def test_ext():
    ext = "cowboy"
    assert acceptable_file(ext) == False

def test_correct_ext():
    ext = "perl"
    assert clean_ext(ext) == "pl"

def test_reg_use():
    reg_expr = "\s__\w*__\("
    prog = re.compile(reg_expr)
    text ='''import packlag
def __init__(self):
    var = thing'''
    val = prog.findall(text)
    print(val)
    assert len(val) == 1

    reg_expr = "\):[ \t]*\n[ \t]*\w"
    prog = re.compile(reg_expr)
    val = prog.findall(text)
    print(val)
    assert len(val) == 1

    reg_expr = "(^from|^import)\s"
    prog = re.compile(reg_expr)
    val = prog.findall(text)
    print(val)
    assert len(val) == 1

    textjs = '''function noAction() {
    }
    '''
    reg_expr = "function\s*\w*\(\w*[\w\s,]*\)\s*\{"
    prog = re.compile(reg_expr)
    val = prog.findall(textjs)
    print(val)
    assert len(val) == 1
