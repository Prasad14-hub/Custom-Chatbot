#Importing all the required class instances from their respective modules
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

import os
import tempfile
from typing import List
#Using tqdm library for creating progress bars to loops and iterable objects. 
from tqdm import tqdm

def create_llm():
    
    # Create llm
    llm = LlamaCpp(
        streaming = True, #for running output
        model_path="Blog/mistral-7b-instruct-v0.1.Q2_K.gguf",
        temperature=0.3,
        top_p=0.8, 
        verbose=True,
        n_ctx=4096 #Context Length 
    )
    return llm

def create_vector_store(pdf_files: List):
    
    vector_store = None

    if pdf_files:
        text = []
        
        for file in tqdm(pdf_files, desc="Processing files"):
            #Getting the file and check it's extension
            file_extension = os.path.splitext(file.name)[1]
            #Writting the PDF file to temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            #Loading the PDF files using PyPdf library 
            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        #Splitting the file into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(text)

        # Creating embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})

        # Creating vector store and storing document chunks using embedding model
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    return vector_store
    
