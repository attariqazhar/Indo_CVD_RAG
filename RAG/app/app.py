import streamlit as st
import os
import json
from dotenv import load_dotenv

from retrieval_generation_pipeline import hypothesis_pipeline, retrieval_pipeline, generation_pipeline

load_dotenv()
url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

# Get topic list from cvd.json
with open("cvd.json", "r") as f:
    cvd_data = json.load(f)

topic_list = []
for topic in cvd_data:
    topic_list.append(topic["topic"])

# Sort topics alphabetically
topic_list.sort()

# Initialize chat history
with open("session_state.json", "r") as f:
    try:
        st.session_state.messages = json.load(f)
    except json.JSONDecodeError:
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "response": "Halo! Saya adalah chatbot yang siap membantu Anda menjawab pertanyaan seputar penyakit jantung. Silakan ajukan pertanyaan Anda."}
    ]
else:
    # Loop through messages and add them to the chat history
    for message in st.session_state.messages:
        if message["role"] == "ai":
            with st.chat_message("ai"):
                st.markdown(message["response"])

                if "context" in message:
                    with st.expander("🔍 Detail"):
                        st.markdown("**Konteks Terkait:**")

                        for content in message["context"]:
                            # Pastikan newline konten dapat ditampilkan dengan benar
                            content = content.replace("\n", "<br>")
                            st.markdown(f"- {content}", unsafe_allow_html=True)

                        st.markdown("**Sumber:**")
                        for source in message["source"]:
                            st.markdown(f"- {source}", unsafe_allow_html=True)

        else:
            with st.chat_message("user"):
                st.markdown(f"**Topik:** {message['topic']}")
                st.markdown(message["query"])
            

# Display dropdown for llm selection, embedding model selection, and topic selection
# Put them in sidebar and in row layout
st.sidebar.title("Indo Cardiac Chatbot")
topic = st.sidebar.selectbox("Topik", topic_list, index=0, help="Pilih topik yang ingin Anda tanyakan.")
llm_name = st.sidebar.selectbox("LLM", ["llama3.1", "deepseek-v2"])
collection_name = st.sidebar.selectbox("Database", ["cvd_collection_v1", "cvd_collection_v2"])
hyde = st.sidebar.checkbox("HyDE", value=False, help="Sebelum menghasilkan jawaban faktual, AI akan menghasilkan hipotesis terlebih dahulu.")

# React to user input
query = st.chat_input("Tanya seputar penyakit jantung")
if query:
    # Tampilkan pesan pengguna
    with st.chat_message("user"):
        st.markdown(f"**Topik:** {topic}")
        st.markdown(query)

    # Simpan ke riwayat pesan
    st.session_state.messages.append({"role": "user", "query": query, "topic": topic})

    # Tampilkan spinner/loading saat menunggu respons
    with st.chat_message("ai"):
        with st.spinner(""):
            # Tentukan embedding model berdasarkan collection
            if collection_name == "cvd_collection_v1":
                embedding_model_name = "nomic-embed-text"
            elif collection_name == "cvd_collection_v2":
                embedding_model_name = "mxbai-embed-large"

            # Jika HyDE diaktifkan
            hyde_response = None
            if hyde:
                hyde_response = hypothesis_pipeline(llm_name, query)

            # Ambil data dari vektor database
            contents, sources = retrieval_pipeline(url, api_key, collection_name, embedding_model_name, query, topic, hyde_response=hyde_response, return_sources=True)

            # Hilangkan duplikasi hasil
            sources = list(set(sources))

            # Dapatkan respons dari model
            response = generation_pipeline(contents, query, llm_name)

        # Tampilkan respons setelah loading selesai
        st.markdown(response)

        # Tambahkan dropdown/expander info teknis
        with st.expander("🔍 Detail"):
            st.markdown("**Konteks Terkait:**")

            for content in contents:
                # Pastikan newline konten dapat ditampilkan dengan benar
                content = content.replace("\n", "<br>")
                st.markdown(f"- {content}", unsafe_allow_html=True)

            st.markdown("**Sumber:**")
            for source in sources:
                st.markdown(f"- {source}", unsafe_allow_html=True)

    # Simpan ke riwayat sebagai pesan dari AI
    st.session_state.messages.append({"role": "ai", "response": response, "context": contents, "source": sources})

    # Export session state to JSON file
    with open("session_state.json", "w") as f:
        json.dump(st.session_state.messages, f, indent=4)

