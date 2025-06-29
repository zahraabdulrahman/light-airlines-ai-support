import pandas as pd
import numpy as np
import os
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DATA_DIR = "data"  # your data directory

def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

def generate_index(csv_filename, index_filename, combine_cols):
    csv_path = os.path.join(DATA_DIR, csv_filename)
    index_path = os.path.join(DATA_DIR, index_filename)

    df = pd.read_csv(csv_path)
    embed_model = load_embedding_model()

    texts = df[combine_cols].agg(" ".join, axis=1).tolist()
    embeddings = embed_model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, index_path)
    return index, df, embed_model

def load_or_create_index(csv_filename, index_filename, combine_cols):
    csv_path = os.path.join(DATA_DIR, csv_filename)
    index_path = os.path.join(DATA_DIR, index_filename)

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        df = pd.read_csv(csv_path)
        embed_model = load_embedding_model()
        return index, df, embed_model
    else:
        return generate_index(csv_filename, index_filename, combine_cols)
