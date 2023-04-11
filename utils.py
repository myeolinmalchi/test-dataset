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
