import numpy as np
import pandas as pd
import glob

file_ext = {"C": ["gcc", "c", "h"],
            "C#": ["csharp"],
            "Clojure": ["clj", "cljs", "cljs", "edn", "clojure"],
            "Common Lisp": ["sbcl"],
            "Haskell": ["hs", "lhs", "ghc"],
            "Java": ["java", "class", "jar"],
            "Javascript": ["js", "javascript"],
            "OCaml": ["ocaml", "ml"],
            "Perl": ["pl", "pm", "t", "pod", "perl"],
            "PHP": ["php", "phtml", "php4", "php3", "php5", "phps", "hack"],
            "Python": ["py", "pyw", "pyc", "pyo", "pyd", "python3", "Python2"],
            "Ruby": ["rb", "rbw", "jruby", "yarv"],
            "Scala": ["scala"],
            "Scheme": ["scm", "ss", "racket"],
            "Tcl": ["tcl"]}

def read_bench_files():
    files = glob.glob("benchmarksgame/benchmarksgame/bench/*/*.*")
    texts = []
    for file in files:
        ext = get_ext(file.split(".")[-1])
        with open(file) as fh:
            if ext != None:
                texts.append((fh.read(), ext))
    return texts

def get_ext(ext):
    for key, value in file_ext.items():
        if ext in value:
            return key


data = read_bench_files()
data = pd.DataFrame(data, columns = ["code", "language"])
print(data)
