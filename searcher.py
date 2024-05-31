from indexer import Posting, computeWordFrequencies, INVERTED_INDEXES_DIR, CHAMPION_LISTS_DIR
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import time
import numpy as np
from collections import defaultdict
import math

DOC_ID_FILE = 'document_id_map.txt'
NUM_RESULTS = 5
STOPWORDS = set(stopwords.words('english'))
total_doc_count = 0
doc_id_map = {}
dfs = {}

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

    # filter out stopwords only if query without stopwords is greater than at least half of the original query
    filtered_tokens = [token for token in query_tokens if token not in STOPWORDS]

    return filtered_tokens if len(filtered_tokens) >= (len(query_tokens)/2) else query_tokens

def _get_merged_posting_lists(query_tokens):
    # for each query token, get merged posting list from alphabetic inv index
    # remember to convert from tuple to Posting
    # return dictionary where key = token, val = Posting list

    merged_posting_lists = defaultdict(dict)

    for token in query_tokens:
        with open(f"{CHAMPION_LISTS_DIR}/{token[0]}.txt") as f:
            for line in f:
                # example line: token: [df, [postings]]
                line = line.split(": ")
                if line[0] == token:
                    data = eval(line[1])
                    dfs[token] = data[0]
                    postings_list = data[1]
                    merged_posting_lists[token] = {p.docid: p for p in [Posting(*posting) for posting in postings_list]}
                    # print(f'{token} merged list length: {len(merged_posting_lists[token].keys())}')
    # print("_get_merged_posting_lists")
    # for token, posting_list in merged_posting_lists.items():
    #     print(f'{token}: {[str(posting) for posting in posting_list]}')

    return merged_posting_lists

def _calculate_ltc_tf_idf(tf, df, total_doc_count):
    # tf-idf = (1+log(tf)) x lsog(N/df)
    # N: total number of documents in the corpus
    # tf: term frequency of term t in document d
    # df: number of documents that contain term t
    
    return (1 + math.log(tf)) * math.log(total_doc_count/(1+df))

def _compute_query_tfidfs(query_tokens, merged_posting_lists):
    # for each merged postings list, get the df for the token.
    # compute the tf-idf for each token in the query.
    # returns a dictionary of tf-idf values for the query (key = token, val = tfidf)
    
    query_token_freqs = computeWordFrequencies(query_tokens)
    query_tfidfs = {}
    
    for token in merged_posting_lists.keys():
        query_tfidfs[token] = _calculate_ltc_tf_idf(query_token_freqs[token], dfs[token], total_doc_count)

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
        for posting in merged_posting_lists[term].values():
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

    norm_doc_vectors = doc_vectors / doc_norms
    norm_query_vector = query_vector / query_norm

    # print('NORMALIZED')
    # for i, doc_id in enumerate(unique_doc_ids):
    #     print(f'{doc_id}: {norm_doc_vectors[:,i]}')

    return (norm_query_vector, norm_doc_vectors)

def _calculate_cosine_similarities(merged_posting_lists, query_tfidfs):  
    # Create axes for query vector and document vector matrix
    unique_terms = list(merged_posting_lists.keys())  # determines the order of the rows in query vector and doc vector matrix -> [merged_posting_lists tokens]
    unique_doc_ids = sorted({docid for v in merged_posting_lists.values() for docid in v.keys()})
    query_vector, doc_vectors = _construct_normalized_query_and_doc_vectors(unique_terms, unique_doc_ids, merged_posting_lists, query_tfidfs)

    # compute cosine scores for all query and doc vector pairs
    cos_scores = defaultdict(list) # create reversed dictionary that stores {cos_score: [doc_ids]}
    for i, docid in enumerate(unique_doc_ids):
        cos_scores[np.dot(query_vector, doc_vectors[:, i])].append(docid)
        
    # print('_calculate_cosine_similarities')
    # print(cos_scores)
    return cos_scores

def _sort_by_desc_tf(docid_list, merged_postings_list):
    # return the doc_id list by order of decreasing total tf
    
    total_tfs = defaultdict()     # {docid: avg_tf}
    for docid in docid_list: 
        total = 0
        # iterate through postings dict for each token in MPL
        for postings_dict in merged_postings_list.values():
            try:
                p = postings_dict[docid]
                total += p.tf
            except:
                pass
        total_tfs[docid] = total
    
    return [docid for docid, _ in sorted(total_tfs.items(), key=lambda x: x[1], reverse=True)]

def _get_top_results(cos_scores, merged_posting_lists):
    # return top results (doc i)
    top_results = []
    remaining_results = NUM_RESULTS
    for _, docid_list in sorted(cos_scores.items(), key=lambda x: x[0], reverse=True):
        sorted_docid_list = _sort_by_desc_tf(docid_list, merged_posting_lists)
        num_results = len(sorted_docid_list)
        
        if remaining_results >= num_results:
            top_results.extend(sorted_docid_list)
            remaining_results -= num_results
        else:
            top_results.extend(sorted_docid_list[:remaining_results])
            remaining_results -= remaining_results
        
        if remaining_results == 0:
            return top_results
        
# Perform a boolean AND search on the merged token lists (i.e. find intersection of doc IDs)
def _boolean_and_search(query_token_postings):
    query_token_postings = sorted(query_token_postings.values(), key=lambda x: len(x))
    running_intersection: list[Posting] = query_token_postings[0]

    for postings_list in query_token_postings[1:]:
        running_intersection = _get_posting_list_intersection(running_intersection, postings_list)

    return [posting.docid for posting in running_intersection]  

def _handle_query(query):
    query_tokens = _preprocess_query(query)
    merged_posting_lists = _get_merged_posting_lists(query_tokens)  # {token: {docid: Posting}}
    query_tfidfs = _compute_query_tfidfs(query_tokens, merged_posting_lists)

    # cos_scores: {score: [doc_ids]}
    cos_scores = _calculate_cosine_similarities(merged_posting_lists, query_tfidfs)

    # comment out for timing queries, uncomment for getting result set count
    # num_matches = 0
    # for score, docids in cos_scores.items():
    #     if score > 0:
    #         num_matches += len(docids)
    
    # print(f'number of matches: {num_matches}')

    top_results = _get_top_results(cos_scores, merged_posting_lists)
    
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
    

def run_local_search_engine():
    _get_doc_id_map_from_disk()

    while True:
        query = input("Enter query (q: quit): ")
        
        if query == "q":
            return
        else:
            start = time.time()
            urls = _handle_query(query)
            end = time.time()
            print(f"Query processing time: {(end-start)*1000} ms\n")
            _print_results(urls)


def run_web_search_engine(query_request):
    if len(doc_id_map.items()) == 0:
        _get_doc_id_map_from_disk()

    start = time.time()
    urls = _handle_query(query_request)
    end = time.time()
    total_time = f"{(end-start)*1000} ms\n"
    # print(f"Query processing time: {(end-start)*1000} ms\n")
    # _print_results(urls)
    return (urls, total_time)

if __name__ == "__main__":
    run_local_search_engine()