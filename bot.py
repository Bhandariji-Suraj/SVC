import pandas as pd
from nlp_model import get_query_embedding, find_top_matches

def load_data(file_path):
    """
    Load the processed dataset and ensure embeddings are properly formatted.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame with embeddings loaded as lists.
    """
    df = pd.read_csv(file_path)
    df['embeddings'] = df['embeddings'].apply(eval)  # Convert string to list
    return df

if __name__ == "__main__":
    # Path to the processed dataset
    dataset_path = "datasets/processed_data.csv"
    data = load_data(dataset_path)

    print("Welcome to the WebBot! Type your query to find relevant results.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() == "exit":
            print("Exiting the WebBot. Goodbye!")
            break

        # Generate embedding for the query
        query_embedding = get_query_embedding(user_query)

        # Find top matches
        results = find_top_matches(query_embedding, data)

        # Display results
        if results.empty:
            print("No relevant matches found for your query.")
        else:
            print("\nTop Matches:")
            for i, row in results.iterrows():
                print(f"Rank {i + 1}:")
                print(f"Link: {row['Link']}")
                print(f"Summary: {row['Summary']}")
                print(f"Similarity Score: {row['similarity']:.2f}\n")
