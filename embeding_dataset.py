import os

from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader

import pinecone

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = os.environ['PINECONE_ENV']

pdf_dir = './dataset'
files = os.listdir(pdf_dir)

pdf_files = [file for file in files if file.endswith('.pdf')]

# Initialize Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = "test-dataset"

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

pinecone.create_index(
    index_name,
    dimension=1536,
    metric='cosine',
)


def embed_pdf(file_name):
    file_path = os.path.join(pdf_dir, file_name)

    # Use PyPDFLoader to load and split pages from the PDF file
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    # Split pages into chunks for better text representation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)

    # Store the document embeddings in Pinecone
    Pinecone.from_documents(docs, embeddings, index_name=index_name)

for file_name in pdf_files:
    embed_pdf(file_name)
