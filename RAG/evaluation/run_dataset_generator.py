import pandas as pd
import os
from dotenv import load_dotenv
import json
import time

# LLM
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

# Pipeline
import pandas as pd
import sys
from dotenv import load_dotenv
import json
import time

# LLM
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

# Pipeline
pipeline_path = os.path.abspath('../pipeline')
if pipeline_path not in sys.path:
    sys.path.append(pipeline_path)
from retrieval_generation_pipeline import hypothesis_pipeline, retrieval_pipeline, generation_pipeline

def load_evaluation_dataset(file_path):
    """
    Args:
        file_path (str): Path to the evaluation dataset CSV file.
    
    Returns:
        evaluation_dataset (pd.DataFrame): DataFrame containing the evaluation dataset.
    """

    # Mengambil data dari file CSV
    evaluation_dataset = pd.read_csv(file_path)

    return evaluation_dataset

def generate_answer_dataset(url, api_key, collection_name, embedding_model_name, llm_name, eval_dataset, hyde=False):
    '''
    Args:
        url (str): URL of the Qdrant server
        api_key (str): API key for Qdrant
        collection_name (str): Name of the collection to search for chunks
        embedding_model_name (str): Name of the model to be used for embeddings
        query (str): Query text
        topic (str): Topic to filter the chunks

    Returns:

    '''
    eval_dict = {
        "question": [],
        "ground_truth": [],
        "retrieved_contexts": [],
        "answer": []
    }

    len_dataset = len(eval_dataset)

    hypothesis = []

    for index, row in eval_dataset.iterrows():
        # Print the progress
        print(f"Row {index + 1}/{len_dataset}", flush=True, end="\r")
        eval_dict["question"].append(row["question"])
        eval_dict["ground_truth"].append(row["ground_truth"])
        query = row["question"]
        topic = row["topic"]
        hyde_response = None
        if hyde:
            hyde_response = hypothesis_pipeline(llm_name, query)
            hypothesis.append(hyde_response)
        retrieved_documents = retrieval_pipeline(url, api_key, collection_name, embedding_model_name, query, topic, hyde_response)
        eval_dict["retrieved_contexts"].append(retrieved_documents)

        llm_answer = generation_pipeline(retrieved_documents, query, llm_name)
        eval_dict["answer"].append(llm_answer)

        # Sleep 30s
        time.sleep(30)
    
    if hyde:
        eval_dict["hypothesis"] = hypothesis
    
    print(f"Answer generation is finished.")

    return eval_dict

def main():
    load_dotenv()
    # Load RAG_config.json
    with open('../../config/eval_configs.json') as f:
        configs = json.load(f)
    
    # Load dataset untuk evaluasi
    eval_dataset = load_evaluation_dataset('../../data/evaluation_dataset.csv')

    # Mengambil URL dan API key untuk terhubung ke Qdrant
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    evaluation_duration = {
        "config": [],
        "duration": []
    }

    for config in configs:
        if config['answer_finished'] == False:
            config_name = config["name"]
            collection_name = config["collection"]
            embedding_model_name = config["embedding_model"]
            llm_name = config["llm_model"]
            is_hyde = config["hyde"]
            print(f"Generating answer for {config_name}")
            print(f"Collection      : {collection_name}")
            print(f"Embedding Model : {embedding_model_name}")
            print(f"LLM Model       : {llm_name}")
            print(f"HyDE            : {is_hyde}\n------------------------------------------------------------")

            start_time = time.time()
            
            answer_json = generate_answer_dataset(url, api_key, collection_name, embedding_model_name, llm_name, eval_dataset, hyde=is_hyde)
            file_name = f"answer_{config_name}.json"
            with open(f'../../data/{file_name}', 'w') as f:
                json.dump(answer_json, f, indent=4)
            
            config['answer_finished'] = True
            with open(f'../../config/eval_configs.json', 'w') as f:
                json.dump(configs, f, indent=4)

            end_time = time.time()

            duration = end_time - start_time

            print(f"Answer saved to '{file_name}'. Duration: {duration:.2f} seconds\n")
            
            # Simpan durasi evaluasi ke dalam dictionary
            evaluation_duration["config"].append(config_name)
            evaluation_duration["duration"].append(duration)  

            # Simpan durasi evaluasi ke dalam file CSV
            duration_df = pd.DataFrame(evaluation_duration)
            duration_df.to_csv(f'../../data/answer_duration_{config_name}.csv', index=False)

main()