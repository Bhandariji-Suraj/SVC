from flask import Flask, request, jsonify, render_template
import pandas as pd
from src.nlp_model import get_query_embedding, find_top_matches

app = Flask(__name__)

# Load dataset
dataset_path = "datasets/processed_data.csv"
try:
    data = pd.read_csv(dataset_path)
    # Ensure the embeddings column is processed as a list
    data['embeddings'] = data['embeddings'].apply(eval)  # Convert string to list
except Exception as e:
    print(f"Error loading dataset: {e}")
    data = pd.DataFrame()  # Fallback to an empty DataFrame if loading fails

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    
    # Validate query input
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    # Get query embedding using the nlp_model
    try:
        query_embedding = get_query_embedding(query)
    except Exception as e:
        return jsonify({"error": f"Error processing query embedding: {str(e)}"}), 500
    
    # Find top matches based on query embedding
    try:
        results = find_top_matches(query_embedding, data)
    except Exception as e:
        return jsonify({"error": f"Error finding top matches: {str(e)}"}), 500
    
    # Prepare the response
    response = [
        {"rank": i + 1, "link": row['Link'], "summary": row['Summary'], "similarity": row['similarity']}
        for i, row in results.iterrows()
    ]
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
