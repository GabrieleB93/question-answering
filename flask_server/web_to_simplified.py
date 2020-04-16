#!/usr/bin/python
# -*- coding: <utf-8> -*-


#TODO: fixare problema di stringhe con o finlandese SOLO NELL'URL
import argparse
import os
import re
from bs4 import BeautifulSoup

def replace_blanks(string):
    string = string.replace(" ", "_")
    return string

def filter_html_tags(string):
    #print(string)
    soup = BeautifulSoup(string, 'html.parser')
    #Keep only body
    body = soup.find('body')
    #Remove all attibutes from html tags and *sanitize html*
    for tag in body.findAll(True):
        tag.attrs = None
    string = str(body)
    #print(string)
    #Delete all html tags except for: br p h1 h2 h3 li ul tr table td
    #OBS: it is case *sensitive*
    line = re.sub(r"<\/?(?!br)(?!p)(?!h1)(?!h2)(?!h3)(?!li)(?!ul)(?!tr)(?!table)(?!td)\w*\b[^>]*>", "", string)
    return line

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

def clean_html(page_html):
    with open(page_html, 'r') as myfile:
        string = myfile.read()
        string = filter_html_tags(string)
        string = replace_blanks(string)
        string = replace_endls_with_blanks(string)
        string = wrap_punctuation_with_blanks(string)
        string = remove_adj_duplicates(string)
        string = replace_back_blanks(string)
    #print(string)
    return string



def create_simplified_input(question_text, page_url, page_html):
    """Under the assumption that html_file is 'clean enough'
    this function creates the data structure of an example"""
    with open('boh.html', 'w+') as html_file:
        html_file.write(page_html)
    page_html_cleaned = clean_html('boh.html')
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