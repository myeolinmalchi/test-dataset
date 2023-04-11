from typing import List
from langchain.schema import Document
from langchain.vectorstores import Pinecone
from langchain.vectorstores.base import VectorStoreRetriever

class CustomVectorStoreRetriever(VectorStoreRetriever):
    vectorstore: Pinecone
    def get_relevant_documents(self, query: str) -> List[Document]:
        if self.search_type == "similarity":
            results = self.vectorstore.similarity_search_with_score(query, **self.search_kwargs)
            for doc, score in results:
                print(f"\n#Score: {score}")
                print(f"{doc.page_content}")
            docs = [result[0] for result in results]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

def create_chunks(sentences, max_chunk_length=1000):
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Add the current sentence to the current chunk
        new_chunk = current_chunk + " " + sentence.strip()
        
        # Check if the new chunk length is within the limit
        if len(new_chunk) <= max_chunk_length:
            current_chunk = new_chunk
        else:
            # If the new chunk exceeds the limit, add the current chunk to the list of chunks
            chunks.append(current_chunk.strip())
            # Start a new chunk with the current sentence
            current_chunk = sentence.strip()

    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
