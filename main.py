import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.vectorstores import FAISS

import os, shutil
from pathlib import Path
import time
import openai
from openai_settings import API_BASE, API_KEY, API_TYPE, API_VERSION, ORGANIZATION, DEPLOYMENT_NAME, MODEL_VERSION

# Settings from GTO. Should be removed for Credit Union usage.
BASE_URL = "https://cgi-openai-gpt4.openai.azure.com/"
API_KEY = "5a9d9a0128e7471eb12ec5efa878d2e4"
DEPLOYMENT_NAME = "gpt-4-32k"
MODEL_VERSION = "2023-07-01-preview"


# BASE_URL = API_BASE
# # os.environ['OPENAI_API_KEY'] = API_KEY
# openai.api_key = API_KEY
# openai.organization = ORGANIZATION

st.set_page_config(page_title="CreditUnion-Assistant")

original_title = '<p style="font-family:Courier; color:#2c1a5f; font-size: 24px;">CreditUnion-Assistant</p>'
st.markdown(original_title, unsafe_allow_html=True)

second_title = ('<p style="font-family:Courier; color:#2c1a5f; font-size: 20px;">'
                'Welcome to CreditUnion help portal, a place where members get answers to all questions related to '
                'CreditUnion services. </p>')
st.markdown(second_title, unsafe_allow_html=True)

pdfs_base = "KnowledgeBase_PDF/"
rebuild_vectordb = False
local_vector_dir = Path('./pdf_vector_store/')
file_name = "faiss_index"
path_to_vectordb = local_vector_dir.joinpath(file_name)


@st.cache_resource(ttl=None)
def configure_retriever(uploaded_files, rebuild_vectordb=True):
    vectordb = get_vectordb(uploaded_files, path_to_vectordb, rebuild_vectordb=rebuild_vectordb)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    return retriever


def get_vectordb(uploaded_files, path_to_vectordb, rebuild_vectordb=True):
    # Read documents
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = None
    if rebuild_vectordb:
        docs = []
        temp_dir = pdfs_base
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir, file)
            loader = PyPDFLoader(temp_filepath)
            docs.extend(loader.load())

        # Split documents
        text_splitter = get_splitter('char_split')(chunk_size=1500, chunk_overlap=200)
        vector_doc_splits = text_splitter.split_documents(docs)
        vectordb = FAISS.from_documents(vector_doc_splits, embeddings)
        vectordb.save_local(path_to_vectordb)
    else:
        vectordb = FAISS.load_local(path_to_vectordb, embeddings)

    return vectordb


def get_splitter(splitter='char_split'):
    splitters = {
        'char_split': CharacterTextSplitter,
        'recur_char_split': RecursiveCharacterTextSplitter
    }
    return splitters[splitter]


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


retriever = configure_retriever(os.listdir(pdfs_base), rebuild_vectordb)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version=MODEL_VERSION,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type=API_TYPE,
    temperature=0,
    streaming=True
)

# llm = ChatOpenAI(
#     model_name="gpt-3.5-turbo", openai_api_key=openai.api_key, temperature=0, streaming=True
# )

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
