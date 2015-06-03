import os
import re


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

            if ext != '.ocaml':
                with open(name, 'r') as fh:
                    programs[0].append(fh.read())
                    programs[1].append(ext)

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
