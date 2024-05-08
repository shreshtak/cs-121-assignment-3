import os
from bs4 import BeautifulSoup
import json
from nltk.tokenize import word_tokenize
from collections import defaultdict

# INVERTED INDEX STRUCTURE
# {string: [postings]}
#   posting tuple: (doc_id, term_freq)
inverted_index = defaultdict(list)
doc_id_map = {} # key = integer id, value  = file path

def computeWordFrequencies(tokens: list):
    freqs = {}

    for token in tokens:
        if token in freqs:
            freqs[token] += 1   # increment count if token already seen
        else:
            freqs[token] = 1    # initialize count of new token to 1

    return freqs

def _tokenize_file(file_path):
    tokens = None

    with open(file_path) as f:
        for line in f:
            # extract json
            line_contents = json.loads(line)
            # print(file_contents)

            url = line_contents['url']
            content = line_contents['content']
            encoding = line_contents['encoding']

            # beautiful soup the content
            soup = BeautifulSoup(content, features='xml')

            # tokenize the page content from soup
            tokens = word_tokenize(soup.get_text())
        
    return tokens

def create_inverted_index():
    # Assign directory
    doc_id_counter = 0
    
    directory = "DEV"
    
    # Iterate over files in directory
    for path, folders, _ in os.walk(directory):
        for folder_name in folders:
            for file in os.listdir(f"{path}/{folder_name}"):
                file_path = f"{path}/{folder_name}/{file}"
                print(file_path)

                #file_type = ""
                # if '.xml' in file_path, file_type = 'lxml'
                # elif '
                # create doc id for document
                doc_id_map[doc_id_counter] = file_path
                
                # tokenize file's HTML content
                tokens = _tokenize_file(file_path)
                # print(tokens)
                
                # compute word frequencies of tokens
                freqs = computeWordFrequencies(tokens)

                # add to inverted index
                for token, freq in freqs.items():
                    inverted_index[token].append((doc_id_counter, freq))
                
                doc_id_counter += 1

                # TO-D0: write to disk, get total size (in KB) of your index on disk
                
                # break   # REMOVE - FOR TESTING
            # break   # REMOVE - FOR TESTING
        break

if __name__ == '__main__':
    create_inverted_index()
    print(inverted_index)

