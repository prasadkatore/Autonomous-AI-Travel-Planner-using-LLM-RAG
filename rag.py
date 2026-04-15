from sentence_transformers import SentenceTransformer
import faiss
import os

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load documents
def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

# Create vector store
def create_vector_store(docs):
    embeddings = model.encode(docs)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, docs

# Retrieve relevant documents
def retrieve(query, index, docs):
    query_embedding = model.encode([query])
    _, indices = index.search(query_embedding, k=1)
    return docs[indices[0][0]]