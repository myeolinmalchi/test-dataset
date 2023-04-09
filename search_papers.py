import os
import pinecone
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
import sys

user_question = sys.argv[1]

if len(sys.argv) != 2:
    sys.exit()

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = os.environ['PINECONE_ENV']

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = "test-dataset"

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

index = pinecone.Index(index_name)

def recommend_papers(question: str, top_k: int = 5):
    # Embed the user's question
    question_embedding = embeddings.embed_query(question)

    # Fetch the top_k most similar documents from Pinecone
    top_results = index.query(
        top_k=top_k*2, 
        include_values=False, 
        include_metadata=True, 
        vector=question_embedding,
    )

    results = [(result.id, result.score, result.metadata) for result in top_results['matches']]

    recommended_papers = []
    seen_paper_names = set()
    for paper_id, similarity, metadata in results:
        # Extract the original document's identifier from the paper_id
        file_name = metadata["source"]

        if file_name not in seen_paper_names:
            seen_paper_names.add(file_name)
            recommended_papers.append((paper_id, similarity, metadata))

        # Stop adding recommendations when top_k unique papers have been added
        if len(recommended_papers) == top_k:
            break

    return recommended_papers

# Example usage
recommended_papers = recommend_papers(user_question)

for paper_id, similarity, metadata in recommended_papers:
    print(f"Paper ID: {paper_id}, Similarity: {similarity:.4f}, Filename: {metadata['source']}")

