from indexer import Posting, computeWordFrequencies
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import time
import numpy as np
import math
from collections import defaultdict

DOC_ID_FILE = 'document_id_map.txt'
INVERTED_INDEXES_DIR = "inverted_indexes"
NUM_RESULTS = 5

total_doc_count = 0
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
    global total_doc_count
    
    with open(DOC_ID_FILE, 'r') as f:
        for line in f:
            id, url = line.split(': ')
            doc_id_map[int(id)] = url.rstrip()
            total_doc_count += 1

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
    # return dictionary where key = token, val = Posting list

    merged_posting_lists = defaultdict(list)

    for token in query_tokens:
        with open(f"{INVERTED_INDEXES_DIR}/{token[0]}.txt") as f:
            for line in f:
                line = line.split(": ")
                if line[0] == token:
                    merged_posting_lists[token] = [Posting(*p) for p in eval(line[1])]
       
    # print("_get_merged_posting_lists")
    # for token, posting_list in merged_posting_lists.items():
    #     print(f'{token}: {[str(posting) for posting in posting_list]}')

    return merged_posting_lists

def _calculate_tf_idf(tf, df):
    # tf-idf = (1+log(tf)) x lsog(N/df)
    # N: total number of documents in the corpus
    # tf: term frequency of term t in document d
    # df: number of documents that contain term t
    
    return (1 + math.log(tf)) * math.log(total_doc_count/(1+df))

def _compute_query_and_doc_tfidfs(query_tokens, merged_posting_lists):
    # for each merged postings list, get the df for the token.
    # update the tf-idf for each Posting and compute the tf-idf for each token in the query.
    # returns a dictionary of tf-idf values for the query (key = token, val = tfidf)
    
    query_token_freqs = computeWordFrequencies(query_tokens)
    query_tfidfs = {}
    
    for token, postings in merged_posting_lists.items():
        df = len(postings)
        for p in postings:
            p.tfidf = _calculate_tf_idf(p.tf, df)

        query_tfidfs[token] = _calculate_tf_idf(query_token_freqs[token], df)

    # print("_compute_query_and_doc_tfidfs")
    # for token, posting_list in merged_posting_lists.items():
    #     print(f'{token}: {[str(posting) for posting in posting_list]}')

    # print(query_tfidfs)

    return query_tfidfs

def _construct_normalized_query_and_doc_vectors(unique_terms, unique_doc_ids, merged_posting_lists, query_tfidfs):
    # Constructs and returns a query vector and document vector matrix where each entry is the normalized tfidf score for
    # that term in the query or a particular document. The order of the terms on the 0 axis (rows) is determined by
    # the order of the terms in unique_terms, and the order of the docids on the 1 axis (columns) is determined by
    # unique_doc_ids.

    doc_vectors = np.empty((len(unique_terms), len(unique_doc_ids)))
    query_vector = np.empty(len(unique_terms))

    for i, term in enumerate(unique_terms):
        row = dict.fromkeys(unique_doc_ids, 0)
        for posting in merged_posting_lists[term]:
            row[posting.docid] = posting.tfidf
        
        # transform term_vector from dictionary (key = docid, val = tfidf) to array of tfidf vals in ascending order of docid.
        # sort by ascending docid and then map [(docid, tfidf)] to [tfidf]
        doc_vectors[i] = [tfidf for _, tfidf in sorted(row.items(), key=lambda x: x[0])]

        # add tfidf for term in query to query_vector
        query_vector[i] = query_tfidfs[term]

    # normalize query and document vectors
    doc_norms = np.linalg.norm(doc_vectors, 2, axis=0, keepdims=True)
    query_norm = np.linalg.norm(query_vector, 2)

    # print('_construct_normalized_query_and_doc_vectors')
    # print(doc_norms)
    # print(query_norm)
    # print(doc_vectors)
    # print(query_vector)
    # for i, doc_id in enumerate(unique_doc_ids):
    #     print(f'{doc_id}: {doc_vectors[:,i]}')

    doc_norms[doc_norms == 0] = 1  # avoid division by zero
    query_norm = query_norm if query_norm != 0 else 1   # avoid division by zero

    # norm_doc_vectors = doc_vectors / doc_norms
    # norm_query_vector = query_vector / query_norm
    norm_doc_vectors = doc_vectors
    norm_query_vector = query_vector

    # print('NORMALIZED')
    # for i, doc_id in enumerate(unique_doc_ids):
    #     print(f'{doc_id}: {norm_doc_vectors[:,i]}')


    return (norm_query_vector, norm_doc_vectors)

def _calculate_cosine_similarities(merged_posting_lists, query_tfidfs):  
    # Create axes for query vector and document vector matrix
    unique_terms = list(merged_posting_lists.keys())  # determines the order of the rows in query vector and doc vector matrix
    # unique_doc_ids = set()

    # for posting_list in merged_posting_lists.values():
    #     for posting in posting_list:
    #         unique_doc_ids.add(posting.docid)

    # unique_doc_ids = sorted(unique_doc_ids) # unique_doc_ids is now a list of all unique docids in ascending order of docid
    unique_doc_ids = sorted({posting.docid for postings in merged_posting_lists.values() for posting in postings})

    query_vector, doc_vectors = _construct_normalized_query_and_doc_vectors(unique_terms, unique_doc_ids, merged_posting_lists, query_tfidfs)

    # Compute cosine scores for all query and doc vector pairs
    cos_scores = {}
    for i, docid in enumerate(unique_doc_ids):
        cos_scores[docid] = np.dot(query_vector, doc_vectors[:, i])

    # print('_calculate_cosine_similarities')
    # print(cos_scores)
    return cos_scores

# Perform a boolean AND search on the merged token lists (i.e. find intersection of doc IDs)
def _boolean_and_search(query_token_postings):
    query_token_postings = sorted(query_token_postings.values(), key=lambda x: len(x))
    running_intersection: list[Posting] = query_token_postings[0]

    for postings_list in query_token_postings[1:]:
        running_intersection = _get_posting_list_intersection(running_intersection, postings_list)

    return [posting.docid for posting in running_intersection]  


def _handle_query(query):
    query_tokens = _preprocess_query(query)
    merged_posting_lists = _get_merged_posting_lists(query_tokens)
    # print(merged_posting_lists)
    query_tfidfs = _compute_query_and_doc_tfidfs(query_tokens, merged_posting_lists)
    cos_scores = _calculate_cosine_similarities(merged_posting_lists, query_tfidfs)

    # extract top NUM_RESULTS docids from cos_scores
    # top_results = [docid for docid, _ in sorted(cos_scores.items(), key=lambda x: x[1])[:NUM_RESULTS]]
    # print('_handle_query')
    # print(sorted(cos_scores.items(), key=lambda x: x[1], reverse=True))
    top_results = [docid for docid, _ in sorted(cos_scores.items(), key=lambda x: x[1], reverse=True)[:NUM_RESULTS]]


    # BOOL SEARCH: REMOVE WHEN DONE WITH RANKED RETRIEVAL
    # top_results = _boolean_and_search(merged_posting_lists)
    
    # look up urls from doc id file
    urls = []
    for doc_id in top_results:
        urls.append(doc_id_map[doc_id])

    return urls

def _print_results(urls):
    for i, url in enumerate(urls[:5]):
        print(f"({i+1}): {url}\n")
    

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
            print(f"Query processing time: {(end-start)*1000} ms\n")
            _print_results(urls)

if __name__ == "__main__":
    run_search_engine()