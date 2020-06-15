#!/usr/bin/python
# -*- coding: <utf-8> -*-


#TODO: fixare problema di stringhe con o finlandese SOLO NELL'URL
import argparse
import os
import re
from bs4 import BeautifulSoup, Comment


def clean_wiki_page(string):
    
    soup = BeautifulSoup(string, 'html.parser')
    body_content = soup.find("div", {"id": "mw-content-text"})
    # remove nav bar (empty elements with id "toc")
    body_content.find("div", {"id": "toc"}).decompose()
    # remove references
    try:
        body_content.find("div", {"class": "reflist columns references-column-width"}).decompose()
    except:
        print("in this page we didn't find the references")
    # remove "edit" links from all sections
    try:
        body_content.find("span", {"class": "mw-editsection"}).decompose()
    except:
        print("in this page we didn't find the edit part")

    # remove html comments
    for element in body_content(text=lambda text: isinstance(text, Comment)):
        element.extract()

    # remove all empty elements
    for x in body_content.find_all():
        if len(x.get_text(strip=True)) == 0:
            x.extract()

    # remove scripts
    children = body_content.findChildren(recursive=False)[0]
    # create list of essential tags
    cleaned_elements  = children.findChildren(recursive=False)
    # remove trailing stuff from "references" onwards
    from_where_to_drop = 0 
    for el in cleaned_elements:
        if el.find("span", {"id": "See_also"}) or el.find("span", {"id":"References"}):
            break
        from_where_to_drop += 1



    cleaned_elements = cleaned_elements[0:from_where_to_drop]
    
    #remove attributes
    for tag in cleaned_elements:
        tag.attrs = None

    # create string of cleaned tags
    cleaned_str = ""
    for el in cleaned_elements:
        cleaned_str += str(el)
    return cleaned_str

def filter_html_tags(string):
    '''
    soup = BeautifulSoup(string, 'html.parser')
    #Keep only body
    body = soup.find('body')
    #Remove all attibutes from html tags and *sanitize html*
    for tag in body.findAll(True):
        tag.attrs = None
    string = str(body)
   '''
    #Delete all html tags except for: br p h1 h2 h3 li ul tr table td
    #OBS: it is case *sensitive*
    line = re.sub(r"<\/?(?!br)(?!p)(?!h1)(?!h2)(?!h3)(?!li)(?!ul)(?!tr)(?!table)(?!td)\w*\b[^>]*>", "", string)
    #remove [edit] and references
    line = re.sub(r"\[edit\]","",line)
    line = re.sub(r"\[\d+\]","",line)
    return line

def replace_blanks(string):
    string = string.replace(" ", "_")
    return string

def wrap_punctuation_with_blanks(string):
    string = string.replace(";", "_;_")
    string = string.replace(",", "_,_")
    string = string.replace(".", "_._")
    string = string.replace(":", "_:_")
    string = string.replace(")", "_)_")
    string = string.replace("(", "_(_")
    #TODO: ce ne sono altri da aggiungere?
    return string

def replace_endls_with_blanks(string):
    string = string.replace("\n", "_") 
    return string

def remove_adj_duplicates(s, ch="_"):
    new_str = [] 
    l = len(s) 
      
    for i in range(len(s)): 
        if (s[i] == ch and i != (l-1) and
           i != 0 and s[i + 1] != ch and s[i-1] != ch): 
            new_str.append(s[i]) 
              
        elif s[i] == ch: 
            if ((i != (l-1) and s[i + 1] == ch) and
               (i != 0 and s[i-1] != ch)): 
                new_str.append(s[i]) 
                  
        else: 
            new_str.append(s[i]) 
          
    return ("".join(i for i in new_str)) 
    

def replace_back_blanks(string):
    string = string.replace("_", " ")
    return string

def clean_html(page_html, name = None):
    with open(page_html, 'r', encoding="utf-8") as myfile:
        string = myfile.read()
        string = clean_wiki_page(string)
        string = filter_html_tags(string)
        string = replace_blanks(string)
        string = replace_endls_with_blanks(string)
        string = wrap_punctuation_with_blanks(string)
        string = remove_adj_duplicates(string)
        string = replace_back_blanks(string)
    #print(string)
    if name:
        with open(name + ".html", 'w', encoding="utf-8") as outfile:
            outfile.write(string)
    return string



def create_simplified_input(question_text, page_url, page_html, name):
    """Under the assumption that html_file is 'clean enough'
    this function creates the data structure of an example"""
    with open('boh.html', 'w+', encoding="utf-8") as html_file:
        html_file.write(page_html)
    page_html_cleaned = clean_html('boh.html', name = name)
    simplified_input = {
      "question_text": question_text,
      "example_id": 42,
      "document_url": page_url,
      "document_text": page_html_cleaned,
      "long_answer_candidates": [],
      "annotations": []
    }
    return simplified_input

"""
parser = argparse.ArgumentParser()
parser.add_argument('--question_file', type=str, default="question_gemma.txt")
parser.add_argument('--page_url', type=str, default="https://en.wikipedia.org/wiki/Schrodinger%27s_cat")
parser.add_argument('--page_html', type=str, default="source_page_gemma_small.html")#check_remove.txt")
args, _ = parser.parse_known_args()


with open(args.question_file, 'r') as myfile:
    question_text = myfile.read()
print(create_simplified_input(question_text, args.page_url, args.page_html))"""