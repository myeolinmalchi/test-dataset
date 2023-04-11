import os
import time

from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from kss import split_sentences

import pinecone

from utils import create_chunks

def main():
    load_dotenv()

    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
    PINECONE_ENV = os.environ['PINECONE_ENV']

    # Get List of .pdf files
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

    print("Check index exists ... ", end="")

    index = pinecone.Index(index_name = index_name)
    if index_name in pinecone.list_indexes():
        index.delete(delete_all=True)
        print("OK")
    else:
        print("OK")
        print(f"Create index[name:{index_name}] ... ", end="")
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine',
        )
        print("OK")


    # 이거 안하면 오류 발생 왜??
    time.sleep(1)

    # Embed and Store pdf file
    def embed_pdf(idx, file_name):
        file_path = os.path.join(pdf_dir, file_name)

        # Use PyPDFLoader to load and split pages from the PDF file
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        temp = split_sentences(
            text=[page.page_content for page in pages], 
            strip=True, 
        )
        texts = sum(temp, [])
        chuncks = create_chunks(texts)

        '''
        # Split pages into chunks for better text representation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        docs = text_splitter.split_documents(pages)
        '''

        # Store the document embeddings in Pinecone
        Pinecone.from_texts(
            chuncks, 
            embeddings, 
            index_name=index_name, 
            metadatas=[{"file_name": f"{idx+1}. {file_name}"} for _ in range(len(chuncks))]
        )

    for idx, file_name in enumerate(pdf_files):
        print(f"Embedding[{idx+1}/{len(pdf_files)}] ... ", end="")
        embed_pdf(idx, file_name)
        print("OK")

    print("Complete.")

if __name__ == '__main__':
    main()
