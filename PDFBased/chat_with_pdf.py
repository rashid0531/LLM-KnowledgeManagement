import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pathlib import Path
import os
import openai
from openai_settings import API_BASE, API_KEY, API_TYPE, API_VERSION, ORGANIZATION

st.set_page_config(page_title="CMA - GenAI-BOT", page_icon='./CGI_compressed_logo.png')
st.markdown("<h1 style='text-align: center;'>CMA - GenAI-BOT</h1>", unsafe_allow_html=True)

def configure_retriever(docs, create_local_vectordb=False, local_vector_dir=None):
    # Split documents
    text_splitter = CharacterTextSplitter(
                        separator='\n', 
                        chunk_size=1000, 
                        chunk_overlap=200
                    )
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = OpenAIEmbeddings()

    if create_local_vectordb:
        file_name = "faiss_index"
        path_to_vectordb = local_vector_dir.joinpath(file_name)
        if len(os.listdir(local_vector_dir)) == 0:
            vectordb = FAISS.from_documents(docs, embeddings)
            vectordb.save_local(path_to_vectordb)
        else:
            vectordb = FAISS.load_local(path_to_vectordb, embeddings)
    else:
        vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    return retriever

def load_pdfs(pdf_dataset_directory):
    loader = PyPDFDirectoryLoader(pdf_dataset_directory)
    pages = loader.load()
    return pages


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx}, page : {doc.metadata['page']} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


openai.api_key = os.getenv('OPENAI_API_KEY')
openai.organization = ORGANIZATION

pdf_dataset_directory =  r'./KnowledgeBase_PDF/'
local_save_directory = Path('./pdf_vector_store/')

data = load_pdfs(pdf_dataset_directory)
retriever = configure_retriever(data, create_local_vectordb=True, local_vector_dir=local_save_directory)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key=openai.api_key, temperature=0, streaming=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
        print('Source: ', response["source_documents"])