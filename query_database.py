import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
# from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Diberikan konteks berikut:

{context}

---

Jawab pertanyaan ini sesuai dengan konteks yang diberikan: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = OllamaEmbeddings(
        model="mxbai-embed-large"
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=3)
    if len(results) == 0:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # print(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = OllamaLLM(model="llama3.2")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # model = 
    # response_text = model.predict(prompt)

    # sources = [doc.metadata.get("source", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)


if __name__ == "__main__":
    main()