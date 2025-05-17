# # query_rag.py
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

# ------------------- 1. Load Environment -------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY is not set in your .env file.")

# ------------------- 2. Load and Prepare Data -------------------
csv_path = "flipkart_laptop_cleaned_structured.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ File not found: {csv_path}")

df = pd.read_csv(csv_path)

# Create a descriptive text field for each laptop
df["doc"] = df.apply(lambda row: f"Laptop: {row['Product_name']}, CPU: {row['CPU']}, RAM: {row['RAM']}, "
                                 f"Storage: {row['Storage']}, Price: ₹{row['Prices']}, "
                                 f"Battery: {row.get('Battery', 'N/A')}, Description: {row['Description']}",
                    axis=1)

# ------------------- 3. Generate Embeddings -------------------
print("🔗 Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
docs = df["doc"].tolist()
embeddings = model.encode(docs, show_progress_bar=True)

# ------------------- 4. Initialize FAISS -------------------
embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings))
documents = docs.copy()

# ------------------- 5. Initialize Gemini LLM -------------------
# from langchain_google_genai import ChatGoogleGenerativeAI

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# gemini_llm = ChatGoogleGenerativeAI(
#     model="models/chat-bison-001",  # ✅ Make sure you're using a supported model
#     api_key=GOOGLE_API_KEY,
#     temperature=0.7
# )

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

# ------------------- 6. Query Handler -------------------
def handle_query(query, user_name, previous_queries):
    if "history" in query.lower() or "previous searches" in query.lower():
        return f"📚 Your previous search queries were: {', '.join(previous_queries)}" if previous_queries else "You haven't searched anything yet."

    # Create query embedding
    query_embedding = model.encode([query])[0]

    # Search in FAISS index
    if index.ntotal > 0:
        D, I = index.search(np.array([query_embedding]), 5)
        matched_docs = [documents[i] for i in I[0] if i < len(documents)]
    else:
        matched_docs = []

    if not matched_docs:
        return "😕 Sorry, I couldn't find relevant laptops. Try rephrasing your query."

    context = "\n".join(matched_docs)
    prompt = f"""
You are an expert assistant helping {user_name} find the best laptop based on their needs.
Previous searches: {', '.join(previous_queries)}

Context:
{context}

User query: "{query}"

Provide a detailed and relevant recommendation.
"""

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ]
    response = groq_llm.invoke(messages)
    return response.content.strip()

# ------------------- 7. Run Sample Query -------------------
if __name__ == "__main__":
    user_name = "Alex"
    previous_queries = ["gaming laptop", "2-in-1 laptop"]
    query = "I need a laptop with great battery life under ₹40,000"

    response = handle_query(query, user_name, previous_queries)
    print("\n💬 GROQ Response:\n")
    print(response)
