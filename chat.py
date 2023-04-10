from langchain import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
from langchain.vectorstores.base import VectorStoreRetriever
import pinecone
import os
import sys

user_question = sys.argv[1]

if len(sys.argv) != 2:
    sys.exit()

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = os.environ['PINECONE_ENV']

CONDENSE_PROMPT = PromptTemplate.from_template(
'''
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:''')

QA_PROMPT = PromptTemplate.from_template(
'''
You are an AI assistant providing helpful advice. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided.
You should only provide hyperlinks that reference the context below. Do NOT make up hyperlinks.
If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

Question: {question}
=========
{context}
=========
Answer in Markdown:''')

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
index_name = "test-dataset"
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

index = pinecone.Index(index_name)
embedded_question = embeddings.embed_query(user_question)
# Recommend Papers by initail question
def recommend_papers(top_k: int = 5):
    # Fetch the top_k most similar documents from Pinecone
    top_results = index.query(
        top_k=top_k*2, 
        include_values=False, 
        include_metadata=True, 
        vector=embedded_question,
    )

    results = [(result.id, result.score, result.metadata) for result in top_results['matches']]

    recommended_papers = []
    seen_paper_names = set()
    for paper_id, similarity, metadata in results:
        # Extract the original document's identifier from the paper_id
        file_name = metadata["file_name"]

        if file_name not in seen_paper_names:
            seen_paper_names.add(file_name)
            recommended_papers.append((paper_id, similarity, metadata))

        # Stop adding recommendations when top_k unique papers have been added
        if len(recommended_papers) == top_k:
            break

    return recommended_papers

recommended_papers = recommend_papers()
for idx, paper in enumerate(recommended_papers):
    print(f"{idx+1}. {paper[2]['file_name']}")

selected_paper = int(input("\n상세한 질문을 원하는 문서의 번호를 입력하세요: "))

file_name = recommended_papers[selected_paper-1][2]['file_name']

def make_chain(vectorstore): 
    question_generator = LLMChain(
        llm=ChatOpenAI(), 
        prompt=CONDENSE_PROMPT
    )

    doc_chain = load_qa_chain(
        llm=ChatOpenAI(model_name='gpt-4'), 
        prompt=QA_PROMPT
    )

    return ConversationalRetrievalChain(
        retriever=VectorStoreRetriever(vectorstore=vectorstore, search_kwargs={"filter": {"file_name": file_name}}), 
        combine_docs_chain=doc_chain, 
        question_generator=question_generator, 
        return_source_documents=True, 
    )


vectorstore = Pinecone.from_existing_index(
    embedding=embeddings, 
    index_name="test-dataset", 
)

chat_history = []
chain = make_chain(vectorstore)

while True:
    query = input("\n질문: ")
    result = chain({"question": query, "chat_history": chat_history})
    print(f"\n답변:\n{result['answer']}")
    chat_history.append((query, result['answer']))