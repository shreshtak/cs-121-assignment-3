import os
from bs4 import BeautifulSoup
import json
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
from heapq import heappop, heappush

BATCH_SIZE = 2000     # number of documents per partial inverted index
DATA_DIR = "DEV"
INDEXES_DIR = "indexes"


# INVERTED INDEX STRUCTURE
# {string: [postings]}
#   posting tuple: (doc_id, tf_idf)
inverted_index = defaultdict(list)  # DEPRECATED; FOR MILESTONE 1 
doc_id_map = {} # key = integer id, value  = file path

class Posting:
    def __init__(self, docid, tfidf):
        self.docid = docid
        self.tfidf = tfidf  # use freq counts for now

    def __str__(self):
        # return f'({self.docid}, {self.tfidf})'
        return f'({self.docid}, {self.tfidf})'

# returns dictionary {token: frequency}
def computeWordFrequencies(tokens: list):
    freqs = {}

    for token in tokens:
        if token in freqs:
            freqs[token] += 1   # increment count if token already seen
        else:
            freqs[token] = 1    # initialize count of new token to 1

    return freqs

# returns the url and tokens 
def _tokenize_file(file_path):
    tokens = None

    with open(file_path) as f:
        for line in f:
            # extract json
            line_contents = json.loads(line)

            url = line_contents['url']
            content = line_contents['content']

            # beautiful soup the content
            # soup = BeautifulSoup(content, features='xml')
            soup = BeautifulSoup(content, 'html.parser')

            # tokenize the page content from soup
            tokens = word_tokenize(soup.get_text())
        
    return url, tokens


def create_inverted_index():

    # TO DO:
    # - COSINE SIMILARITY METHOD FOR TF-IDF

    doc_id_counter = 0
    
    # create partial index and doc counter to create index of size [BATCH_SIZE]
    partial_inv_index = defaultdict(list)
    doc_batch_counter = 0
    num_batches = 0     
    
    ps = PorterStemmer()
    # Iterate over subfolders in directory
    for path, folders, _ in os.walk(DATA_DIR):
        for folder_name in folders:
            for file in os.listdir(f"{path}/{folder_name}"):
            
                file_path = f"{path}/{folder_name}/{file}"
                print(file_path)   
                             
                # tokenize file's HTML content
                url, tokens = _tokenize_file(file_path)
                tokens = [ps.stem(token) for token in tokens]
                # print(tokens)

                # create doc id for document
                # doc_id_map[doc_id_counter] = url
                # doc_id_counter += 1
                
                # compute word frequencies of tokens
                freqs = computeWordFrequencies(tokens)
                doc_id_map[doc_id_counter] = (url, file_path, freqs)

                # add to partial inverted index
                for token, freq in freqs.items():
                    partial_inv_index[token].append(Posting(doc_id_counter, freq))
                
                # add to main inverted index for testing purposes - REMOVE BEFORE SUBMITTING
                for token, freq in freqs.items():
                    inverted_index[token].append(Posting(doc_id_counter, freq))

                doc_id_counter += 1
                doc_batch_counter += 1

                if doc_batch_counter == BATCH_SIZE:
                    num_batches += 1

                    # write the partial index to disk in alphabetical order of tokens
                    # reset doc counter and continue going through files
                    
                    # write partial_inverted_index to a file
                    os.makedirs(INDEXES_DIR, exist_ok=True)   # create indexes folder if it doesn't exist

                    with open(f'{INDEXES_DIR}/inverted_index{num_batches}.txt', 'w') as f:
                        for token, postings in sorted(partial_inv_index.items(), key=lambda x: x[0]):
                            f.write(f'{token}: {[eval(str(posting)) for posting in postings]}\n')
                    
                    # reset counter and index
                    partial_inv_index = defaultdict(list)
                    doc_batch_counter = 0
                           
           # break   # REMOVE - FOR TESTING
           
    # # write main inverted index to disk - REMOVE LATER
    # with open(f'inverted_index.txt', 'w') as f:
    #     for token, postings in sorted(inverted_index.items(), key=lambda x: x[0]):
    #         f.write(f'{token}: {[eval(str(posting)) for posting in postings]}\n')

    # # write results to file - REMOVE BEFORE SUBMITTING
    # with open('milestone1_report2.txt', 'w') as f:
    #     # get number of documents from doc_id_counter
    #     f.write(f"Number of documents: {doc_id_counter}\n")
        
    #     # get number of unique tokens from len(inverted_index.keys())
    #     f.write(f"Number of unique tokens: {len(inverted_index.keys())}\n")

    #     # get total size (in KB) of file using os.path.getsize()
    #     size = os.path.getsize('inverted_index.txt') / 1000    
    #     f.write(f"Size of the inverted index (in KB): {size}")
    
    
def write_doc_id_map_to_disk():
    with open(f'document_id_map.txt', 'w') as f:
        for doc_id, path in doc_id_map.items():
            f.write(f'{doc_id}: {path}\n')
            

if __name__ == '__main__':
    create_inverted_index()
    # print(inverted_index)
    write_doc_id_map_to_disk()

