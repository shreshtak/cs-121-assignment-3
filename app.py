from flask import Flask, render_template, request
from searcher.py import run_search_engine
 
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', user_input="", results = "")

@app.route('/query', methods=['POST'])
def get_results():
    user_input = request.form['user_input']
    res = run_search_engine(user_input)[:5]
    # res = user_input.split(" ")
    return render_template('index.html', user_input = user_input, results = res)

if __name__ == '__main__':  
   app.run(debug=True)