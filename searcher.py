from indexer import *
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import time
import numpy

DOC_ID_FILE = 'document_id_map.txt'
INVERTED_INDEXES_DIR = "inverted_indexes"

doc_id_map = {}

def _get_posting_list_intersection(token1_postings, token2_postings):
    
    t1_index = 0
    t2_index = 0
    intersection = []

    num_t1_postings = len(token1_postings)
    num_t2_postings = len(token2_postings)

    while t1_index < num_t1_postings and t2_index < num_t2_postings:
        if token1_postings[t1_index].docid < token2_postings[t2_index].docid:
            t1_index += 1
        elif token2_postings[t2_index].docid < token1_postings[t1_index].docid:
            t2_index += 1
        else:
            intersection.append(token2_postings[t2_index])
            t1_index += 1
            t2_index += 1
    
    return intersection

# loads document id map from disk into global instance
def _get_doc_id_map_from_disk():
    with open(DOC_ID_FILE, 'r') as f:
        for line in f:
            id, url = line.split(': ')
            doc_id_map[int(id)] = url.rstrip()


# Do any preprocessing to the query text (tokenizing, capitalization, stemming, etc)
def _preprocess_query(query):
    query_tokens = word_tokenize(query)
    # use porter stemming on tokens
    ps = PorterStemmer()
    query_tokens = [ps.stem(token) for token in query_tokens]
    return query_tokens

def _get_merged_posting_lists(query_tokens):
    # for each query token, get merged posting list from alphabetic inv index
    # remember to convert from tuple to Posting
    # return list of postings
    # query optimization: return postings list in ascending order of length

    posting_lists = []

    for token in query_tokens:
        with open(f"{INVERTED_INDEXES_DIR}/{token[0]}.txt") as f:
            for line in f:
                line = line.split(": ")
                if line[0] == token:
                    posting_lists.append([Posting(*p) for p in eval(line[1])])

    return sorted(posting_lists, key=lambda x: len(x))
        

# Perform a boolean AND search on the merged token lists (i.e. find intersection of doc IDs)
def _boolean_and_search(query_token_postings):
    running_intersection: list[Posting] = query_token_postings[0]

    for postings_list in query_token_postings[1:]:
        running_intersection = _get_posting_list_intersection(running_intersection, postings_list)

    return [posting.docid for posting in running_intersection]  


def _print_results(urls):
    for i, url in enumerate(urls[:5]):
        print(f"({i+1}): {url}\n")

def _handle_query(query):
    query_tokens = _preprocess_query(query)
    merged_posting_lists = _get_merged_posting_lists(query_tokens)
    _calculate_cosine_similarity(query_tokens)
    bool_and_search_results = _boolean_and_search(merged_posting_lists)
    
    # look up urls from doc id file
    urls = []
    for doc_id in bool_and_search_results:
        urls.append(doc_id_map[doc_id])

    return urls

def _calculate_cosine_similarity(query_tokens):
    # 1. Represent the query as a weighted tf-idf vector
    # 2. Represent each document as a weighted tf-idf vector
    # 3. Compute the cosine similarity score for the query vector and each document vector
    # 4. Rank documents with respect to the query by score
    # 5. Return the top K (e.g., K = 10) to the user
    pass

def run_search_engine():
    _get_doc_id_map_from_disk()

    while True:
        query = input("Enter query (q: quit): ")
        # timer start here
        start = time.time()
        if query == "q":
            return
        else:
            urls = _handle_query(query)
            # end timer here
            end = time.time()
            print(f"Query processing time: {(end-start)*1000} ms")
            _print_results(urls)

if __name__ == "__main__":
    run_search_engine()