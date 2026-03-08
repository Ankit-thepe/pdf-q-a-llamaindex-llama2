import os
import tempfile
import streamlit as st
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index import set_global_service_context
from llama_index.embeddings import GradientEmbedding
from llama_index.llms import GradientBaseModelLLM

@st.cache_resource
def build_service_context():
    llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400)
    embed_model = GradientEmbedding(
        gradient_access_token=os.environ.get("GRADIENT_ACCESS_TOKEN", ""),
        gradient_workspace_id=os.environ.get("GRADIENT_WORKSPACE_ID", ""),
        gradient_model_slug="bge-large")
    svc = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, chunk_size=256)
    set_global_service_context(svc)
    return svc

def main():
    st.set_page_config(page_title="Chat with your PDF using Llama2 & Llama Index", page_icon="🦙")
    st.header("🦙 Chat with your PDF using Llama2 model & Llama Index")

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    with st.sidebar:
        st.subheader("Upload Your PDF File")
        docs = st.file_uploader("⬆️ Upload your PDF & Click to process",
                                accept_multiple_files=False, type=["pdf"])
        if st.button("Process") and docs:
            with st.spinner("Processing..."):
                service_context = build_service_context()
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_file = os.path.join(tmp_dir, docs.name)
                    with open(tmp_file, "wb") as f:
                        f.write(docs.getbuffer())
                    documents = SimpleDirectoryReader(tmp_dir).load_data()
                    index = VectorStoreIndex.from_documents(documents,
                                                            service_context=service_context)
                    st.session_state.query_engine = index.as_query_engine()
                    st.session_state.activate_chat = True
            st.success("✅ PDF processed! Ask your questions.")

    if st.session_state.activate_chat:
        if prompt := st.chat_input("Ask your question from the PDF?"):
            with st.chat_message("user", avatar="👨🏻"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "avatar": "👨🏻", "content": prompt})

            pdf_response = st.session_state.query_engine.query(prompt)
            cleaned_response = pdf_response.response
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(cleaned_response)
            st.session_state.messages.append({"role": "assistant", "avatar": "🤖", "content": cleaned_response})
    else:
        st.info("⬅️ Upload a PDF from the sidebar to start chatting.")

if __name__ == "__main__":
    main()
