import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re

# Load CSV
df = pd.read_csv("flipkart_laptop_model.csv")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Create searchable text for each row
df["doc"] = df.apply(lambda row: f"{row['Product_name']} {row['CPU']} {row['RAM']} {row['Storage']} ₹{row['Prices']} {row['Description']}", axis=1)
df["doc"] = df["doc"].apply(clean_text)

# Initialize SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
docs = df["doc"].tolist()
embeddings = model.encode(docs, show_progress_bar=True)

# Save embeddings to FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index and cleaned docs
faiss.write_index(index, "laptop_index.faiss")
df.to_csv("laptop_docs.csv", index=False)

print("✅ Embeddings created and saved successfully!")

