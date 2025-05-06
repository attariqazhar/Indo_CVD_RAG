import pandas as pd
import sys
import os
from dotenv import load_dotenv
from datasets import Dataset
import json

# RAG Evaluation (RAGAS)
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# LLM
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings


# RAG EVALUATION

def evaluate_answers(config_name, llm_name, embedding_name, hyde=False):
    """
    Args:
        config_name (str): Name of the configuration to be evaluated.
        llm_name (str): Name of the LLM model to be used for evaluation.
        embedding_name (str): Name of the embedding model to be used for evaluation.
        hyde (bool): Whether to use HyDE or not.
    
    Returns:
        None
    """

    # Load jawaban dari file JSON
    file_path = f'../../data/answer_{config_name}.json'
    with open(file_path) as f:
        answer_dataset = json.load(f)

    load_dotenv()
    # Instansiasi LLM dan embeddings
    llm = ChatOpenAI(model=llm_name, openai_api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_name, openai_api_key=os.getenv("OPENAI_API_KEY")))
    
    # Evaluasi jawaban
    eval_results = {
        "question": [],
        "ground_truth": [],
        "retrieved_contexts": [],
        "answer": []
    }

    if hyde:
        eval_results["hypothesis"] = []

    eval_results["faithfulness"] = []
    eval_results["answer_relevancy"] = []
    eval_results["context_precision"] = []
    eval_results["context_recall"] = []

    # Mengambil informasi index terakhir dari hasil evaluasi sebelumnya
    # Jika file CSV tidak ada, maka mulai dari index 0
    # Hal ini agar evaluasi tidak terulang dari awal jika sudah pernah dievaluasi sebelumnya
    try:
        df = pd.read_csv(f'.../../data/evaluation_result_{config_name}.csv')
        last_idx = len(df)
    except FileNotFoundError:
        last_idx = 0

    # Ubah jawaban menjadi format Dataset dari Hugging Face
    for i in range(len(answer_dataset["answer"])):
        if i >= last_idx:
            # Print the progress
            print(f"Evaluating row {i + 1}/{len(answer_dataset['answer'])}", flush=True, end="\r")
            answer_row = {
                "question": [answer_dataset["question"][i]],
                "ground_truth": [answer_dataset["ground_truth"][i]],
                "retrieved_contexts": [answer_dataset["retrieved_contexts"][i]],
                "answer": [answer_dataset["answer"][i]],
            }

            if hyde:
                answer_row["hypothesis"] = [answer_dataset["hypothesis"][i]]

            # Melakukan penilaian terhadap jawaban dan konteks yang diambil
            dataset = Dataset.from_dict(answer_row)
            score = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                ],
                llm=llm,
                embeddings=embeddings,
            )

            # Memasukkan hasil evaluasi ke dalam eval_results
            eval_results["question"].append(answer_dataset["question"][i])
            eval_results["ground_truth"].append(answer_dataset["ground_truth"][i])
            eval_results["retrieved_contexts"].append(answer_dataset["retrieved_contexts"][i])
            eval_results["answer"].append(answer_dataset["answer"][i])

            if hyde:
                eval_results["hypothesis"].append(answer_dataset["hypothesis"][i])
            
            eval_results["faithfulness"].append(score['faithfulness'][0])
            eval_results["answer_relevancy"].append(score['answer_relevancy'][0])
            eval_results["context_precision"].append(score['context_precision'][0])
            eval_results["context_recall"].append(score['context_recall'][0])

            # Simpan hasil evaluasi ke dalam file CSV
            # Hal ini dilakukan tiap iterasi untuk menghindari kehilangan data jika terjadi kesalahan
            df = pd.DataFrame(eval_results)

            df = df.applymap(lambda x: str(x).replace('\n', '\\n'))

            df.to_csv(f'../../data/evaluation_result_{config_name}.csv', index=False, sep=',')

def main():
    load_dotenv()
    # Load RAG_config.json
    with open('../../config/eval_configs.json') as f:
        configs = json.load(f)

    # Instansiasi nama LLM dan embeddings
    llm_eval = "gpt-4o-mini"
    embedding_eval = "text-embedding-3-small"

    for config in configs:
        if config['eval_finished'] == False:
            config_name = config["name"]
            collection_name = config["collection"]
            embedding_model_name = config["embedding_model"]
            llm_name = config["llm_model"]
            is_hyde = config["hyde"]
            print(f"Evaluating {config_name}")
            print(f"Collection      : {collection_name}")
            print(f"Embedding Model : {embedding_model_name}")
            print(f"LLM Model       : {llm_name}")
            print(f"HyDE            : {is_hyde}\n------------------------------------------------------------")

            evaluate_answers(config_name, llm_eval, embedding_eval, hyde=is_hyde)
            
            file_name = f"evaluation_result_{config_name}.csv"
            
            config['eval_finished'] = True
            with open(f'../../config/eval_configs.json', 'w') as f:
                json.dump(configs, f, indent=4)

            print(f"Evaluation result saved to '{file_name}'.\n")

main()