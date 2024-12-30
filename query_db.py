import argparse
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEndpoint
# from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from getpass import getpass
import os
from dotenv import load_dotenv
from langchain_huggingface.llms import HuggingFacePipeline

# Memuat file .env
load_dotenv()

# Mengambil token dari .env
# HUGGINGFACEHUB_API_TOKEN = getpass()
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

CHROMA_PATH = "chroma"

# Prompt template untuk jawaban akhir
FINAL_PROMPT_TEMPLATE = """
Diberikan konteks berikut:

{context}

---

Jawab pertanyaan ini sesuai dengan konteks yang diberikan: {question}
"""

# Prompt template untuk jawaban hipotesis (HyDE)
HYDE_PROMPT_TEMPLATE = """
Jawab pertanyaan berikut dengan penjelasan singkat dan umum:

Pertanyaan: {question}

Jawaban:
"""

def main():
    # Membuat CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Teks pertanyaan.")
    args = parser.parse_args()
    query_text = args.query_text

    # Siapkan LLM
    # hyde_model = OllamaLLM(model="Supa-AI/gemma2-9b-cpt-sahabatai-v1-instruct:q3_k_l")
    # model = OllamaLLM(model="llama3.2")
    # hyde_model = HuggingFaceEndpoint(
    #     repo_id='kalisai/Nusantara-2.7b-Indo-Chat',
    #     huggingfacehub_api_token=huggingface_api_token
    # )

    hyde_model = HuggingFacePipeline.from_model_id(
        model_id="kalisai/Nusantara-2.7b-Indo-Chat",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 100}
    )

    # model = HuggingFaceEndpoint(
    #     repo_id='meta-llama/Llama-3.2-3B-Instruct',
    #     huggingfacehub_api_token=huggingface_api_token
    # )
    model = OllamaLLM(model="llama3.2")

    # Langkah 1: Menghasilkan jawaban hipotesis menggunakan LLM
    hyde_prompt_template = ChatPromptTemplate.from_template(HYDE_PROMPT_TEMPLATE)
    hyde_prompt = hyde_prompt_template.format(question=query_text)
    hypothetical_answer = hyde_model.invoke(hyde_prompt).strip()
    print(f"Jawaban Hipotesis:\n{hypothetical_answer}\n")

    # Siapkan fungsi embedding
    embedding_function = OllamaEmbeddings(
        model="mxbai-embed-large"
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Langkah 2: Menggunakan jawaban hipotesis untuk mencari dalam database vektor
    results = db.similarity_search_with_score(hypothetical_answer, k=3)
    if len(results) == 0:
        print(f"Tidak dapat menemukan hasil yang cocok.")
        return

    # Menggabungkan konteks dari dokumen yang ditemukan
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Membuat prompt untuk jawaban akhir
    prompt_template = ChatPromptTemplate.from_template(FINAL_PROMPT_TEMPLATE)
    final_prompt = prompt_template.format(context=context_text, question=query_text)
    print(f"Prompt untuk Jawaban Akhir:\n{final_prompt}\n")

    # Langkah 3: Menghasilkan jawaban akhir menggunakan LLM
    response_text = model.invoke(final_prompt).strip()

    # Mengambil sumber dokumen
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response:\n{response_text}\n\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
