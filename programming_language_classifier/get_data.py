import os
import sys
import pandas as pd

extensions = {".gcc": "C",
              ".c": "C",
              ".csharp": "C#",
              ".sbcl": "Common Lisp",
              ".clojure": "Clojure",
              ".ghc": "Haskell",
              ".java": "Java",
              ".javascript": "JavaScript",
              ".js": "JavaScript",
              ".ocaml": "OCaml",
              ".perl": "Perl",
              ".hack": "PHP",
              ".php": "PHP",
              ".py": "Python",
              ".python3": "Python",
              ".jruby": "Ruby",
              ".yarv": "Ruby",
              ".scala": "Scala",
              ".racket": "Scheme",
              ".tcl": "TCL"}

def get_content(directory):
    content = []
    for file in os.listdir(directory):
        extension = os.path.splitext(file)[1]
        if extension in extensions:
            with open(directory + file) as fh:
                content.append([extensions[extension], fh.read()])
    return content


def make_dataframe(content_list):
    return pd.DataFrame(content_list)


if __name__ == '__main__':
    content_list = get_content(sys.argv[1])
    print(make_dataframe(content_list))