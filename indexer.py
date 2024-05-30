import os
from bs4 import BeautifulSoup
import json
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
from string import ascii_lowercase
import heapq
from simhash import Simhash, SimhashIndex

BATCH_SIZE = 2000     # number of documents per partial inverted index
DATA_DIR = "DEV"
PARTIAL_INDEXES_DIR = "partial_indexes"
INVERTED_INDEXES_DIR = "inverted_indexes"
WEIGHTED_TAGS = ['h1', 'h2', 'h3', 'b', 'strong']
ALNUM_KEYS= list(ascii_lowercase) + [str(i) for i in range(10)]

doc_id_map = {} # key = integer id, value  = file path
simhash_index = SimhashIndex([], k=1)

class Posting:
    def __init__(self, docid, tf, tfidf=None):
        self.docid = docid
        self.tf = tf  # use freq counts for now
        self.tfidf = tfidf

    def __str__(self):
        return f'({self.docid}, {self.tf}, {self.tfidf})'  # posting tuple when eval-ed: (doc_id, tf, tfidf)

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
            soup = BeautifulSoup(content, 'html.parser')

            weighted_tag_content = ' '.join([tag.string for tag in soup.findAll(WEIGHTED_TAGS, string=True)])

            # tokenize the page content from soup
            tokens = word_tokenize(soup.get_text(' ') + ' ' + weighted_tag_content)
        
    return url, tokens


def create_partial_indexes():

    # TO DO:
    # - COSINE SIMILARITY METHOD FOR TF-IDF

    doc_id_counter = 0
    
    # create partial index and doc counter to create index of size [BATCH_SIZE]
    # INVERTED INDEX STRUCTURE: {string: [postings]}
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

                # checks for near duplicates by simhashing the file's tokens
                simhash = Simhash(tokens)
                near_duplicates = simhash_index.get_near_dups(simhash) # finds existing pages that are near duplicates, returns count
                if (near_duplicates > 0): # if the duplicate count is greater than 0, don't proceed
                    continue
                simhash_index.append(tokens, simhash) # else add it to simhash_index

                # create doc id for document
                doc_id_map[doc_id_counter] = url
                
                # compute word frequencies of tokens
                freqs = computeWordFrequencies(tokens)

                # add to partial inverted index
                for token, freq in freqs.items():
                    if token[0] in ALNUM_KEYS:
                        partial_inv_index[token].append(Posting(doc_id_counter, freq))

                doc_id_counter += 1
                doc_batch_counter += 1

                if doc_batch_counter == BATCH_SIZE:
                    num_batches += 1

                    # write the partial index to disk in alphabetical order of tokens
                    # reset doc counter and continue going through files
                    
                    # write partial_inverted_index to a file
                    os.makedirs(PARTIAL_INDEXES_DIR, exist_ok=True)   # create partial indexes folder if it doesn't exist

                    with open(f'{PARTIAL_INDEXES_DIR}/partial_index{num_batches}.txt', 'w') as f:
                        for token, postings in sorted(partial_inv_index.items(), key=lambda x: x[0]):
                            f.write(f'{token}: {[eval(str(posting)) for posting in postings]}\n')
                    
                    # reset counter and index
                    partial_inv_index = defaultdict(list)
                    doc_batch_counter = 0    
       
def write_doc_id_map_to_disk():
    with open(f'document_id_map.txt', 'w') as f:
        for doc_id, path in doc_id_map.items():
            f.write(f'{doc_id}: {path}\n')

# Get and return the merged postings list for each of the query tokens
def merge_indexes():
    inverted_index_files = {}

    partial_index_files = []
    num_partial_index_files = 0

    partial_index_lines = {}    # to store the ((token, i), posting list) pairs to be merged
    priority_queue = []
    global priority_queue_length
    priority_queue_length = 0

    def get_full_sorted_posting_list_of_next_token_and_write_to_disk():
        global priority_queue_length

        # get first (token, i) from priority queue
        same_token = [heapq.heappop(priority_queue)]       # stores the (token, i) pairings from the priority queue
        priority_queue_length -= 1

        current_token = same_token[0][0]
        print(f'current token is "{current_token}"')
        # continue popping (token, i) pairs from priority queue while tokens are the same
        while True:
            if priority_queue_length == 0 or priority_queue[0][0] != current_token:
                break
            
            elem = heapq.heappop(priority_queue)
            priority_queue_length -= 1

            same_token.append(elem)
        
        # for all the docs that contain the token, pop the entries from partial_index_lines and merge the posting lists together
        full_postings_list = []
        for token in same_token:
            full_postings_list.extend(partial_index_lines.pop(token))

        # # if token doesn't start with alnum character, don't bother sorting
        # # we do this after popping from priority_queue and partial_index_lines so that they don't build up
        # if current_token[0].lower() not in ALNUM_KEYS:
        #     return
        
        # sort the full postings list
        full_postings_list.sort()
        
        # store token and its merged list in appropriate alnum inverted index file
        file = inverted_index_files[current_token[0]]
        file.write(f"{current_token}: {full_postings_list}\n")
        print(f'Writing "{token}" to {file.name}')   

        # END OF HELPER FUNCTION
    
    # open all the inverted index files and store them in a dictionary with alnum char as the key
    os.makedirs(INVERTED_INDEXES_DIR, exist_ok=True)   # create inverted indexes folder if it doesn't exist

    for alnum in ALNUM_KEYS:
        inverted_index_files[alnum] = open(f"{INVERTED_INDEXES_DIR}/{alnum}.txt", 'w')
    
    # open all the partial index files and store them in an array
    for partial_index in os.listdir(PARTIAL_INDEXES_DIR):
        partial_index_files.append(open(f'{PARTIAL_INDEXES_DIR}/{partial_index}', 'r'))
        num_partial_index_files += 1

    # execute loop while there are still partial index files to read
    while True:
        non_empty_line_read = False
        for i, partial_index_file in enumerate(partial_index_files):
            # read line and separate into token and posting list
            line = partial_index_file.readline()
            
            if line:
                non_empty_line_read = True
                line = line.split(': ')
                token = line[0]

                heapq.heappush(priority_queue, (token, i))
                priority_queue_length += 1

                #eval the postings list and put it in partial_index_lines with (token, i) as key
                partial_index_lines[(token,i)] = eval(line[1].strip())

        if not non_empty_line_read:
            # all partial index files are done reading
            break

        # continue if priority_queue is empty
        if priority_queue_length < 1:
            continue
       
        get_full_sorted_posting_list_of_next_token_and_write_to_disk()

    # process any remaining tokens in priority_queue
    while priority_queue_length > 0:
        get_full_sorted_posting_list_of_next_token_and_write_to_disk()


if __name__ == '__main__':
    create_partial_indexes()
    write_doc_id_map_to_disk()
    merge_indexes()
