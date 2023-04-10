import os
import time

from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
from kss import split_sentences

import pinecone

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

    # Embed and Store pdf file
    def embed_pdf(file_name):
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

        # Split pages into chunks for better text representation
        '''
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
        )
        '''
        #docs = text_splitter.split_documents(pages)
        #docs = text_splitter.create_documents(chuncks)

        # Store the document embeddings in Pinecone
        Pinecone.from_texts(
            chuncks, 
            embeddings, 
            index_name=index_name, 
            metadatas=[{"file_name": file_name} for _ in range(len(chuncks))]
        )

    for idx, file_name in enumerate(pdf_files):
        print(f"Embedding[{idx+1}/{len(pdf_files)}] ... ", end="")
        embed_pdf(file_name)
        print("OK")

    print("Complete.")

if __name__ == '__main__':
    main()
