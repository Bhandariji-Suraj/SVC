import pandas as pd
from sentence_transformers import SentenceTransformer

def generate_embeddings(input_file, output_file):
    """
    Generate embeddings for the summary column in the dataset.

    Args:
        input_file: Path to the input CSV file containing the dataset.
        output_file: Path to save the processed dataset with embeddings.
    """
    # Load the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Read the dataset
    df = pd.read_csv(input_file)

    # Generate embeddings for each summary
    if 'Summary' not in df.columns:
        raise ValueError("The input dataset must contain a 'summary' column.")
    df['embeddings'] = df['Summary'].fillna("").apply(lambda x: model.encode(x).tolist())

    # Save the processed dataset
    df.to_csv(output_file, index=False)
    print(f"Embeddings added and saved to {output_file}")

if __name__ == "__main__":
    input_path = "datasets/summary_df.csv"
    output_path = "datasets/processed_data.csv"
    generate_embeddings(input_path, output_path)
