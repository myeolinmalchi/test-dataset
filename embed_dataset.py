import os
import time

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
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

    chat = ChatOpenAI(model="gpt-3.5-turbo")
    def extract_cpc(text):
        system_message = SystemMessage(content="You are a helpful AI assistant.")
        user_message = HumanMessage(content=f'''
        Here is a summary of the patent. 
        Based on the summary, guess the CPC of the patent and answer in a json array.
        Please do not include explanations and quotation marks in your answer.
        The CPC is only guessed for the major classification, not the minor classification.
        Try to guess as many CPCs as possible.
        
        Wrong Answer: ['F03D 13/10', 'F03D 11/00']
        Correct Answer: ['F03D', 'F03D']

        Summary:
        {text}''')

        answer = chat.generate([[system_message, user_message]])
        return answer.generations[0][0].text

    # Embed and Store pdf file
    def embed_pdf(idx, file_name):
        file_path = os.path.join(pdf_dir, file_name)

        # Use PyPDFLoader to load and split pages from the PDF file
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        temp = split_sentences(
            text=[page.page_content for page in pages], 
            strip=False, 
        )

        texts = sum(temp, [])

        chuncks = create_chunks(texts)

        cpc_string = extract_cpc(chuncks[0])
        cpc_list = eval(cpc_string)

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
            metadatas=[{"file_name": f"{idx+1}. {file_name}", "cpc": cpc_list} for _ in range(len(chuncks))]
        )

    for idx, file_name in enumerate(pdf_files):
        print(f"Embedding[{idx+1}/{len(pdf_files)}] ... ", end="")
        embed_pdf(idx, file_name)
        print("OK")

    print("Complete.")

if __name__ == '__main__':
    main()
