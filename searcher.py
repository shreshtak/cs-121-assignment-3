from indexer import *
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

DOC_ID_FILE = 'document_id_map.txt'

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
            id, url = line.split(': ', maxsplit=1)
            doc_id_map[int(id)] = eval(url.rstrip())


# Do any preprocessing to the query text (tokenizing, capitalization, stemming, etc)
def preprocess_query(query):
    query_tokens = word_tokenize(query)
    # use porter stemming on tokens
    ps = PorterStemmer()
    query_tokens = [ps.stem(token) for token in query_tokens]
    return query_tokens

# Get and return the merged postings list for each of the query tokens
def merge_indexes(query_tokens):
    query_token_postings = defaultdict(list)    # {query_: [postings]}

    # get postings lists for each token ()
    #   merge postings lists from the partial indexes into one large inverted index for token t
    for file in os.listdir(INDEXES_DIR):
        with open(f'{INDEXES_DIR}/{file}', 'r') as f:
            for line in f:
                token, postings = line.split(': ')

                if token in query_tokens:
                    postings_list_as_tuples = eval(postings)
                    postings_list_as_posting = [Posting(x[0], x[1]) for x in postings_list_as_tuples]
                    query_token_postings[token].extend(postings_list_as_posting)
    
    return query_token_postings


# Perform a boolean AND search on the merged token lists (i.e. find intersection of doc IDs)
def boolean_and_search(query_token_postings):
    query_token_postings = list(query_token_postings.values())
    running_intersection: list[Posting] = query_token_postings[0]

    for postings_list in query_token_postings[1:]:
        running_intersection = _get_posting_list_intersection(running_intersection, postings_list)
    
    return [posting.docid for posting in running_intersection]  

def _print_results(urls):
    for i, url in enumerate(urls[:5]):
        print(f"({i+1}): {url[0]}\n")

def handle_query(query):
    query_tokens = preprocess_query(query)
    merged_posting_lists = merge_indexes(query_tokens)
    bool_and_search_results = boolean_and_search(merged_posting_lists)
    
    # look up urls from doc id file
    urls = []
    for doc_id in bool_and_search_results:
        urls.append(doc_id_map[doc_id])

    _print_results(urls)

def run_search_engine():
    _get_doc_id_map_from_disk()

    while True:
        query = input("Enter query (q: quit): ")
        if query == "q":
            return
        else:
            handle_query(query)


if __name__ == "__main__":
    run_search_engine()