import os
import json
import pandas as pd
from dotenv import load_dotenv
from uuid import uuid4

# Vector DB
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Embedding Model
from langchain_ollama import OllamaEmbeddings


# DOCUMENT PREPARATION
def get_json_document(file_path):
    '''
    Args:
        file_path (str): Path to the JSON file containing the data
    
    Returns:
        data (list): List of dictionaries containing the data from the JSON file
    '''

    with open(file_path) as f:
        data = json.load(f)
    return data

# STORE DOCUMENT TO QDRANT

### Embedding Model
def get_embedding_model(model_name):
    '''
    Args:
        model_name (str): Name of the model to be used for embeddings
    
    Returns:
        embeddings (OllamaEmbeddings): Embedding model object
    '''

    embeddings = OllamaEmbeddings(
        model=model_name,
    )
    return embeddings

def encode_text(embeddings, text):
    '''
    Args:
        embeddings (OllamaEmbeddings): Embedding model object
        text (str): Text to be encoded

    Returns:
        np.array: Encoded vector for the input text
    '''
    
    # Melakukan embedding pada teks
    return embeddings.embed_query(text)

def add_vector(documents, embeddings):
    '''
    Args:
        documents (list): List of dictionaries containing the data
        embeddings (OllamaEmbeddings): embedding model

    Returns:
        documents (list): List of dictionaries containing the data with added vector embedding
    '''

    # Menambahkan vector embedding ke setiap dokumen
    for doc in documents:
        doc['id'] = str(uuid4())
        doc['vector'] = encode_text(embeddings, doc['content'])
    return documents

def get_documents_with_vector(file_path, model_name):
    '''
    Args:
        file_path (str): Path to the JSON file containing the data
        model_name (str): Name of the model to be used for embeddings

    Returns:
        documents (list): List of dictionaries containing the data with added vector embedding
    '''

    documents = get_json_document(file_path)
    embeddings = get_embedding_model(model_name)
    documents = add_vector(documents, embeddings)
    return documents

### Creating Database
def instantiate_database(collection_name, embedding_model):
    '''
    Returns:
        client (QdrantClient): QdrantClient object
        embedding_model (OllamaEmbeddings): Embedding model object
    '''

    # Mengakses Qdrant Client menggunakan API yang sudah dibuat sebelumnya
    load_dotenv(override=True)
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

    # Mengambil ukuran embedding dari model yang digunakan
    embedding_size = len(encode_text(embedding_model, 'test'))
    
    # Membuat collection di Qdrant
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="topic",
        field_schema="keyword"
    )

    return client

### Store Text and Vector to Database
def add_document(client, collection_name, doc):
    '''
    Args:
        client (QdrantVectorStore): Vector store object
        collection_name (str): Name of the collection to store the documents in
        doc: JSON document to be stored
    '''

    # Menyimpan dokumen ke dalam Qdrant
    client.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=doc['id'], vector=doc['vector'], payload={"topic":doc['topic'], "subtopic":doc['subtopic'], "text":doc['content'], "source":doc['url']})]
    )

def store_documents(client, collection_name, documents):
    '''
    Args:
        client (QdrantVectorStore): Vector store object
        collection_name (str): Name of the collection to store the documents in
        documents (list): List of Document objects
    '''

    # Menyimpan kumpulan dokumen ke dalam Qdrant
    for doc in documents:
        add_document(client, collection_name, doc)
    print("Documents stored successfully")

# INDEXING PIPELINE
def indexing_pipeline(file_path, model_name, document_name, collection_name):
    '''
    Args:
        file_path (str): Path to the JSON file containing the data
        model_name (str): Name of the model to be used for embeddings
        document_name (str): Name of the document to be exported
        collection_name (str): Name of the collection to store the documents in

    Returns:
        None
    '''

    documents = get_documents_with_vector(file_path, model_name)

    # Menyimpan backup metadata dokumen (tanpa vector)
    doc_path = os.path.join('../../data/backup', document_name)
    with open(doc_path, 'w') as f:
        json.dump(documents, f, indent=4)
    print("Documents exported to", document_name)

    # Instantiasi database
    client = instantiate_database(collection_name, get_embedding_model(model_name))
    print("Database instantiated successfully")
    
    # Menyimpan dokumen ke dalam database
    store_documents(client, collection_name, documents)



def main():

    # Instantiasi file path yang digunakan untuk menyimpan dokumen
    file_path = '../../data/cvd_prepared.json'

    # Load file konfigurasi untuk Qdrant
    with open('../../config/vectordb_configs.json') as f:
        configs = json.load(f)

    # Melakukan penyimpanan dokumen ke dalam Qdrant
    for config in configs:
        document_name = config['backup_file_name']
        model_name = config['embedding_model']
        collection_name = config['collection']

        print(f"Storing documents to {collection_name} using '{model_name}' model...")

        # Menjalankan pipeline indexing
        indexing_pipeline(file_path, model_name, document_name, collection_name)

main()
