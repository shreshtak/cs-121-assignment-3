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

class Posting:
    def __init__(self, docid, tfidf):
        self.docid = docid
        self.tfidf = tfidf  # use freq counts for now

    def __str__(self):
        # return f'({self.docid}, {self.tfidf})'
        return f'({self.docid}, {self.tfidf})'


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

            content = line_contents['content']

            # beautiful soup the content
            soup = BeautifulSoup(content, features='xml')

            # tokenize the page content from soup
            tokens = word_tokenize(soup.get_text())
        
    return tokens

def create_inverted_index():
    # Assign directory
    doc_id_counter = 0
    
    directory = "DEV"
    
    # Iterate over subfolders in directory
    for path, folders, _ in os.walk(directory):
        for folder_name in folders:
            for file in os.listdir(f"{path}/{folder_name}"):
                file_path = f"{path}/{folder_name}/{file}"
                print(file_path)

                # create doc id for document
                doc_id_map[doc_id_counter] = file_path
                
                # tokenize file's HTML content
                tokens = _tokenize_file(file_path)
                # print(tokens)
                
                # compute word frequencies of tokens
                freqs = computeWordFrequencies(tokens)

                # add to inverted index
                for token, freq in freqs.items():
                    inverted_index[token].append(Posting(doc_id_counter, freq))
                
                doc_id_counter += 1
                
                # break   # REMOVE - FOR TESTING
            # break   # REMOVE - FOR TESTING
        break

    # write inverted_index to a file
    with open('inverted_index.txt', 'w') as f:
        for token, postings in inverted_index.items():
            f.write(f'{token}: {[eval(str(posting)) for posting in postings]}\n')

            # break # REMOVE - FOR TESTING
    
    
    with open('milestone1_report.txt', 'w') as f:
        # get number of documents from doc_id_counter
        f.write(f"Number of documents: {doc_id_counter}\n")
        
        # get number of unique tokens from len(inverted_index.keys())
        f.write(f"Number of unique tokens: {len(inverted_index.keys())}\n")

        # get total size (in KB) of file using os.path.getsize()
        size = os.path.getsize('inverted_index.txt') / 1000    
        f.write(f"Size of the inverted index (in KB): {size}")

if __name__ == '__main__':
    create_inverted_index()
    print(inverted_index)
