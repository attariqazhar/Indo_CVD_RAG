# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import BedrockEmbeddings
# from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
# import chromadb
import nltk

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
# openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def get_embedding_functions():
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large"
    )
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="attariqazhar", region_name="ap-southeast-1", model_id='amazon.titan-embed-text-v1'
    # )
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="bedrock-admin", region_name="us-east-1", model_id='amazon.titan-embed-text-v1'
    # )
    return embeddings

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, get_embedding_functions(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()