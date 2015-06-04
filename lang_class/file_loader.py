import os
import re

names = {'.cs':'C#', '.java':'Java', '.c':'C', '.gcc':'C', '.js':'JavaScipt',
         '.hack':'PHP', '.php':'PHP', '.racket':'Scheme',
         '.sbcl':'Common Lisp', '.jruby':'Ruby', '.yarv':'Ruby', '.rb':'Ruby',
         '.hs':'Haskell', '.lhs':'Haskell', '.ocaml':'Ocaml', '.prl':'Perl',
         '.clojure':'Clojure', '.clj':'Clojure', '.ats':'Ada', '.csharp':'C#',
         '.dart':'Dart', '.erlang':'Erlang', '.fpascal':'Pascal',
         '.fsharp':'F#', '.ghc':'Haskell', '.gnat':'Ada', '.go':'Go',
         '.gpp':'C', '.ifc':'Fortran', '.javascript':'Javascript',
         '.lua':'Lua', '.oz':'Oz', '.perl':'Perl', '.py':'Python',
         '.python3':'Python', '.rust':'Rust', '.scala':'Scala', '.vw':'unknown',
          '.cint':'C', '.javasteady':'Java', '.parrot':'Perl'}


def load_files(name=None):
    """
    Loads all text from programs from a given folder
    and their file extension into a list
    and returns it
    """

    if name:
        files = get_names(name)
    else:
        files = get_names()

    programs = [[],[]]

    for name in files:
        if get_ext(name):
            ext = get_ext(name).group(0)

            with open(name, 'r') as fh:
                programs[0].append(fh.read())
                programs[1].append(names[ext])

    return programs

def get_names(name='../benchmarks'):
    f = []

    filenames = os.walk(name)

    for filename in filenames:
        if filename[1] == []:
            for item in filename[2]:
                f.append(filename[0] + '/' + item)

    return f


def get_ext(filename):
    return re.search(r'\.([\w]+)$', filename)
