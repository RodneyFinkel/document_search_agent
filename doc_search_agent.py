import os
import numpy as np
import pandas as pd
from datasets import load_dataset
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time


dataset = load_dataset("wikipedia", "20220301.en", split="train[:1000]", trust_remote_code=True)
print(f"Loaded {len(dataset)} Wikipedia articles.")

documents = []
for i, article in enumerate(dataset):
    doc = {
        "id": f"doc_{i}",
        "title": article["title"],
        "text": article["text"],
        "url": article["url"]
    }
    documents.append(doc)
    
df = pd.DataFrame(documents)
df.head(3)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

# Split documents into smaller chunks for more granular searches
chunks = []
chunk_ids = []
chunk_sources = []

for i , doc in enumerate(documents):
    doc_chunks = text_splitter.split_text(doc["text"])
    chunks.extend(doc_chunks)
    chunk_ids.extend([f"chunk_{i}_{j}" for j in range(len(doc_chunks))])
    chunk_sources.extend([doc["title"]] * len(doc_chunks))

print(f"Created {len(chunks)} chunks from {len(documents)} documents")

# Create sentence embeddings for the chunks
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(model_name)

# Test the model with a sample text
sample_text = "This is a sample text to test our embedding model."
sample_embedding = embedding_model.encode(sample_text)
print(f"Embedding dimension: {len(sample_embedding)}")

chroma_client = chromadb.Client()
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_name
)

# set up chromad client and create collection
collection = chroma_client.create_collection(
    name="document_search",
    embedding_function=embedding_function
)

batch_size = 100
for i in range(0, len(chunks), batch_size):
    end_idx = min(i + batch_size, len(chunks))
    
    batch_ids = chunk_ids[i:end_idx]
    batch_chunks = chunks[i:end_idx]
    batch_sources = chunk_sources[i:end_idx]
    
    collection.add(
        ids=batch_ids,
        documents=batch_chunks,
        metadatas=[{"source": source} for source in batch_sources],    
    )

    print(f"Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} to the collection")

print(f"Total documents in collection: {collection.count()}")

# Perform a search through the documents similar to the query
# Args: query (str) n_results (int) Returns: dict: Search results
def search_documents(query, n_results=5):
    
    start_time = time.time()
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    end_time = time.time()
    search_time = end_time - start_time
    print(f"Search completed in {search_time:.4f} seconds")
    return results

# Example search
queries = [
    "Evolution of Real Analysis in Mathematics",
    "History of artificial intelligence",
    "Space exploration missions"
]

for query in queries:
    print(f"\nQuery: {query}")
    results = search_documents(query)
    
    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\nResult {i+1} from {metadata['source']}:")
        print(f"{doc[:200]}...")
        
    

