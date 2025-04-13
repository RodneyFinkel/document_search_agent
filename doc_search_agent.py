import os
import numpy as np
import pandas as pd
from datasets import load_dataset
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import asyncio
# from transformers import BertTokenizer, BertModel


dataset = load_dataset("wikipedia", "20220301.en", split="train[:1000]", trust_remote_code=True, num_proc=4)
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
    chunk_size=500,
    chunk_overlap=100,
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
# model_name = "sentence-transformers/all-mpnet-base-v2"  
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

all_metadatas = [{"source": source} for source in chunk_sources]

def add_batch(start_idx, end_idx):
    batch_ids = chunk_ids[start_idx:end_idx]
    batch_chunks = chunks[start_idx:end_idx]
    batch_metadatas = all_metadatas[start_idx:end_idx]
    
    collection.add(
        ids=batch_ids,
        documents=batch_chunks,
        metadatas=batch_metadatas,
        # metadatas=[{"source": source} for source in batch_sources],
    )
    print(f"Added batch {start_idx // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} to the collection")


# batch_size = 100
# #  Use ThreadPoolExecutor to add chunks in parralel
# test_start = time.time()
# with ThreadPoolExecutor() as executor:
# # with ProcessPoolExecutor() as executor:
#     futures = [
#         executor.submit(add_batch, i, min(i + batch_size, len(chunks)))
#         for i in range(0, len(chunks), batch_size)
#     ]
# test_end = time.time()
# test_time = test_end - test_start
# print(f"Batch addition completed in {test_time:.4f} seconds")
# print(f"Total documents in collection: {collection.count()}")   



# test_start = time.time()
# for i in range(0, len(chunks), batch_size):
#     end_idx = min(i + batch_size, len(chunks))
    
#     batch_ids = chunk_ids[i:end_idx]
#     batch_chunks = chunks[i:end_idx]
#     batch_sources = chunk_sources[i:end_idx]
    
#     collection.add(
#         ids=batch_ids,
#         documents=batch_chunks,
#         metadatas=[{"source": source} for source in batch_sources],    
#     )

#     print(f"Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} to the collection")
# test_end = time.time()
# test_time = test_end - test_start
# print(f"Batch addition completed in {test_time:.4f] seconds}")
# print(f"Total documents in collection: {collection.count()}")



# Perform a search through the documents similar to the query
# Args: query (str) n_results (int) Returns: dict: Search results
def filtered_search(query, n_results=5):
    
    # where_clause = {"source": filter_source} if filter_source else None 
    start_time = time.time()
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        # where=where_clause
    )
    
    end_time = time.time()
    search_time = end_time - start_time
    print(f"Search completed in {search_time:.4f} seconds")
    return results
    
    # unique_sources = list(set(chunk_sources))
    # print(f"Available sources for filtering: {len(unique_sources)}")
    # print(unique_sources[:5])

    # if len(unique_sources) > 0:
    #     filter_source =  unique_sources[0]
    #     query = "main concepts and principles"
    #     print(f"\nFiltered search for '{query}' in source '{filter_source}':")
    #     results = filtered_search(query, filter_source=filter_source)
    
    #     for i, doc in enumerate(results['documents'][0]):
    #         print(f"\nResult {i+1}:")
    #         print(f"{doc[:200]}...")
            
    # return results

        
def interactive_search():
    
    while True:
        query = input("\nEnter your search query (or 'quit' to exit):")
        
        if query.lower() == "quit":
            break
            
        n_results = int(input("How many results would you like?"))
        
        results = filtered_search(query, n_results)
        
        print(f"\nFound {len(results['documents'][0])} results for '{query}':")
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
            )):
                relevance = 1 - distance
                print(f"\n--- Result {i+1} ---")
                print(f"Source: {metadata['source']}")
                print(f"URL: {metadata.get('url', 'N/A')}") 
                print(f"Relevance: {relevance:.2f}")
                print(f"Excerpt: {doc[:300]}...")  
                print("-" * 50)
                
                     
# interactive_search()

if __name__ == "__main__":
    
    batch_size = 500
    ##num_threads = os.cpu_count()
    test_start = time.time()
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(add_batch, i, min(i + batch_size, len(chunks)))
            for i in range(0, len(chunks), batch_size)
        ]
    test_end = time.time()
    test_time = test_end - test_start
    print(f"Batch addition completed in {test_time:.4f} seconds")
    print(f"Total documents in collection: {collection.count()}") 
    interactive_search()
    




