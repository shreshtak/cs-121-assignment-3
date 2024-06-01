# cs-121-assignment-3
OIH 14021174
RGGUO 38627512
SHRESHK 64486304
Assignment 3 for CS 121

Make sure the following resources are installed using pip/pip3: bs4, nltk, simhash, numpy, flask


How to Create the Inverted Index and Champion Lists:
1. Make sure your DEV folder is in the project root folder.
2. cd into the project root folder.
3. In the main function at the bottom of indexer.py, make sure create_partial_indexes(), write_doc_id_map_to_disk(), and merge_indexes() are all uncommented. 
    a. If you only want to recreate the inverted index and champion lists without recreating the partial indexes, comment out create_partial_indexes() and write_doc_id_map_to_disk() before running.
4. Run the following command: 'python3 indexer.py' or 'python indexer.py'

How to Run the Local Text Interface Search Engine:
1. Make sure you ran indexer.py (i.e. make sure you created the inverted index and champion lists).
2. Make sure nltk's stopwords are downloaded. See this link: https://pythonspot.com/nltk-stop-words/
2. cd into the project root folder. 
3. Run the following command: 'python3 searcher.py' or 'python searcher.py'
4. Type your query into the terminal when prompted, then hit Enter. Your search results will then be printed.
5. To exit the interface, type 'q' when prompted for a query and hit Enter.

How to Run Web GUI Search Engine:
1. Run this command to install flask:
    'pip3 install flask' or 'pip install flask'
2. Run the following command to start the web GUI:
    'export FLASK_DEBUG=1 && python3 -m flask run'
3. The GUI will be hosted on your computer's local host, for example 'http://127.0.0.1:5000'. 
   Your local host will be linked in the terminal, open it in your browser. 
4. Type your query into the input box and click 'Submit', the search engine results will be returned below
