from googlesearch import search
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import urllib.request

class Page(object):
    query = ""
    url = ""
    content = ""

    def __init__(self, q, u, c):
        self.query = q
        self.url = u
        self.content = c

def remove_stopwords(q):
    #Remove stopwords
    nlp = English()
    my_doc = nlp(q)
    # Create list of word tokens
    token_list = []
    for token in my_doc:
        token_list.append(token.text)

    filtered_query = ""       
    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_query += " " + word
    return filtered_query

def add_wiki(q):
    #Add Wikipedia
    return q + " wikipedia"

def fetch_urls(q):
    my_results_list = []
    for i in search(q,        # The query you want to run
                    tld = 'com',  # The top level domain
                    lang = 'en',  # The language
                    num = 10,     # Number of results per page
                    start = 0,    # First result to retrieve
                    stop = 1,  # Last result to retrieve
                    pause = 2.0,  # Lapse between HTTP requests
                ):
        my_results_list.append(i)
    return my_results_list

def url_to_html(url):
    fp = urllib.request.urlopen(url)
    mybytes = fp.read()

    mystr = mybytes.decode("utf8")
    fp.close()
    return mystr

def query_to_page(q):
    u = fetch_urls(add_wiki(remove_stopwords(q)))[0]
    c = url_to_html(u)
    p = Page(q, u, c)
    return p

"""
query = "where was shrodinger born?"
my_page = query_to_page(query)
print(my_page.query)"""

def obtain_body(page):
    pass
    #only if it is a wikipedia page maybe?
    # use beatifoul soup to -> bs https://www.crummy.com/software/BeautifulSoup/bs4/doc/
    # 1) obtain only the body of the article:
    # <div id="bodyContent" class="mw-body-content"> from this point on 
    # 2) remove navigation box 
    # <div id="toc" class="toc" role="navigation" aria-labelledby="mw-toc-heading">
    # remove everithing from see_also to the end of the file 
    # <span class="mw-headline" id="See_also">See also</span>

