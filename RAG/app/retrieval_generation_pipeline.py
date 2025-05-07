import os
from dotenv import load_dotenv

# Vector DB
from qdrant_client import QdrantClient, models

# Embedding Model
from langchain_ollama import OllamaEmbeddings

# Large Language Model
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
import cohere


# DOCUMENT RETRIEVAL

### Embedding Model
def get_embedding_model(model_name):
    '''
    Args:
        model_name (str): Name of the model to be used for embeddings
    
    Returns:
        embeddings (OllamaEmbeddings): Embedding model object
    '''

    embeddings = OllamaEmbeddings(
        model=model_name
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
    
    return embeddings.embed_query(text)

### Retrieval
def get_chunks(vector, client, collection_name, topic=None):
    '''
    Args:
        query (str): Query text
        embeddings (OllamaEmbeddings): Embedding model object
        client (QdrantClient): Qdrant client object
        collection_name (str): Name of the collection to search for chunks

    Returns:
        chunks (list): list of top 10 chunks from the collection
    '''
    
    # Mengambil 10 chunk teratas dengan similarity tertinggi
    if topic is None:
        chunks = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=10
        )
    else:
        chunks = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=10,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="topic",
                        match=models.MatchAny(any=[topic])
                    )
                ]
            )
        )
    chunks = chunks.points
    return chunks

def get_content_list(chunks, query):
    '''
    Args:
        chunks (list): List of ScoredPoint
        query (str): Query text
    Returns:
        reranked_chunks (list): List of reranked chunks based on query
    '''
    load_dotenv()
    co = cohere.Client(os.getenv("COHERE_API_KEY"))

    chunks = {
        'content': [chunk.payload['text'].replace('Â','') for chunk in chunks],
        'source': [chunk.payload['source'] for chunk in chunks]
    }

    # Melakukan rerank terhadap isi chunk
    reranked_content = co.rerank(
        model="rerank-v3.5", query=query, documents=chunks['content'], top_n=3
    )

    reranked_chunks = {
        'content': [],
        'source': []
    }

    for result in reranked_content.results:
        reranked_chunks['content'].append(chunks['content'][result.index])
        reranked_chunks['source'].append(chunks['source'][result.index])

    return reranked_chunks

# GENERATING RESPONSE

### Formulating a Prompt
def generate_prompt(content_list, query):
    '''
    Args:
        content_list (list): List of content
    Returns:
        str: Prompt for LLM
    '''

    # Membuat prompt untuk LLM menggunakan konteks yang diberikan
    prompt = "Diberikan konteks berikut:\n"
    for num, content in enumerate(content_list):
        prompt += f"{content}"
        prompt += "------------------\n"
    prompt += "Jawab pertanyaan ini selengkap mungkin sesuai dengan konteks yang diberikan: " + query
    # prompt += "Berdasarkan konteks yang diberikan, jawab pertanyaan ini: " + query
    prompt += "\nJangan menyebutkan 'berdasarkan konteks yang diberikan'"
    # prompt += "\nBerikan jawaban dengan mengutip isi konteks yang relevan."
    return prompt

def generate_hyde_prompt(query):
    '''
    Args:
        query (str): Query text
    Returns:
        str: Prompt for LLM
    '''

    # Membuat prompt untuk LLM untuk pipeline RAG + HyDE
    # Prompt ini digunakan agar LLM menjawab pertanyaan dengan pengetahuan yang dimiliki
    prompt = "Kamu adalah seorang ahli penyakit jantung. Jawab pertanyaan ini dengan pengetahuan yang kamu miliki: " + query
    prompt += "\nJangan menyebutkan 'Jawaban pertanyaan tersebut adalah'"
    return prompt

### Generating LLM Response
def get_llm(model_name, from_ollama=True, run_locally=True):
    '''
    Args:
        model_name (str): Name of the model to be used for LLM
        from_ollama (bool): Whether the model is from Ollama or Hugging Face
    Returns:
        llm: LLM model object
    '''

    # Mengambil model LLM dari Ollama atau Hugging Face
    if from_ollama:
        llm = OllamaLLM(model=model_name, temperature=0.5)
    else:
        if run_locally:
            # model_id = model_name
            # tokenizer = AutoTokenizer.from_pretrained(model_id)
            # model = AutoModelForCausalLM.from_pretrained(model_id)
            # pipe = pipeline(
            #     "text-generation", 
            #     model=model, 
            #     tokenizer=tokenizer, 
            #     max_new_tokens=512
            # )
            # llm = HuggingFacePipeline(pipeline=pipe)
            llm = HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text-generation",
                pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.5}
            )
        else:
            llm = HuggingFaceEndpoint(
                repo_id=model_name,
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                max_new_tokens=512,
                temperature=0.5
            )
    return llm

def get_response(model, prompt):
    '''
    Args:
        model (OllamaLLM/HuggingFaceEndpoint): LLM model object
        prompt (str): Prompt for the user
    Returns:
        reponse (str): Response from the LLM model
    '''

    # Mengambil response dari LLM model
    response = model.invoke(prompt).strip()
    return response

# RETRIEVAL AND GENERATION PIPELINE

def hypothesis_pipeline(hyde_model_name, query, from_ollama=True, run_locally=False):
    '''
    Args:
        hyde_model_name (str): Name of the model to be used for HyDE
        query (str): Query text
        from_ollama (bool): Whether the model is from Ollama or Hugging Face
        run_locally (bool): Whether to run the model locally or on Hugging Face
    Returns:
        query (str): Query text
        hyde_response (str): Response from the LLM model
    '''

    # Pipeline ini digunakan untuk sistem RAG + HyDE

    # Mengambil model LLM untuk menghasilkan jawaban hipotesis
    hyde_llm = get_llm(hyde_model_name, from_ollama, run_locally)

    # Menghasilkan prompt untuk LLM HyDE
    hyde_prompt = generate_hyde_prompt(query)

    # Mengambil response dari LLM HyDE
    hyde_response = get_response(hyde_llm, hyde_prompt)

    return hyde_response

def retrieval_pipeline(url, api_key, collection_name, embedding_model_name, query, topic, hyde_response = None, return_sources=False):
    '''
    Args:
        url (str): URL of the Qdrant server
        api_key (str): API key for Qdrant
        collection_name (str): Name of the collection to search for chunks
        embedding_model_name (str): Name of the model to be used for embeddings
        query (str): Query text
        topic (str): Topic to filter the chunks
        return_sources (bool): Whether to return sources along with content
    Returns:
        list: List of content or content and source based on return_sources flag
    '''

    # Get Qdrant client
    client = QdrantClient(url=url, api_key=api_key)

    # Mengambil embedding model
    embeddings = get_embedding_model(embedding_model_name)

    if hyde_response is None:
        # Mengubah teks query menjadi vector embedding
        query_vector = encode_text(embeddings, query)
    else:
        # Mengubah teks hipotesis (untuk HyDE) menjadi vector embedding
        query_vector = encode_text(embeddings, hyde_response)

    # Mengambil 3 chunk teratas dari Qdrant
    chunks = get_chunks(query_vector, client, collection_name, topic=topic)

    # Mengambil isi dan source dari setiap chunk
    contents = get_content_list(chunks, query)

    if return_sources:
        return contents['content'], contents['source']
    else:   
        return contents['content']

def generation_pipeline(content_list, query, llm_name, from_ollama=True, run_locally=False):
    '''
    Args:
        content_list (list): List of content
        query (str): Query text
        llm_name (str): Name of the model to be used for LLM
        from_ollama (bool): Whether the model is from Ollama or Hugging Face
        run_locally (bool): Whether to run the model locally or on Hugging Face
    Returns:
        str: Response from the LLM model
    '''

    # Membuat prompt untuk LLM
    prompt = generate_prompt(content_list, query)

    # Mengambil model LLM dari Ollama atau Hugging Face
    llm = get_llm(llm_name, from_ollama, run_locally)

    # Mengambil response dari LLM model
    response = get_response(llm, prompt)
    return response

