import bs4
import requests
import sys
import re


languages = {'javascript': '.js',
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
             'tcl': '.tcl'}

def rosetta_scraper(seed, path):
    response = requests.get(seed)
    soup = bs4.BeautifulSoup(response.text)
    divs = soup.select("div")
    for div in divs:
        if div.attrs.get("id") and div.attrs.get("id") == "mw-pages":
            all_a = div.select('a')
    links = ["http://rosettacode.org" + a.attrs.get("href")
             for a in all_a
             if a.attrs.get("href") and "wiki" in a.attrs.get("href")]
    count = 1
    for link in links:
        response = requests.get(link)
        soup = bs4.BeautifulSoup(response.text)
        code = soup.select('pre')
        for block in code:
            for key in languages:
                if block.attrs.get('class') is not None and key in block.attrs.get('class'):
                    soup = bs4.BeautifulSoup(re.sub(r'<br/>', "\n", str(block)))
                    with open(path + str(count) + languages[key], "w+") as file:
                        file.write(soup.text)
                        count += 1


if __name__ == '__main__':
    rosetta_scraper(sys.argv[1], sys.argv[2])
