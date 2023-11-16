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
from openai_settings import API_BASE, API_KEY, API_TYPE, API_VERSION

from langchain.chat_models import AzureChatOpenAI

def load_urls(list_of_urls):
    loaders = UnstructuredURLLoader(urls=list_of_urls)
    data = loaders.load()
    return data

def load_pdfs(pdf_dataset_directory):
    loader = PyPDFDirectoryLoader(pdf_dataset_directory)
    pages = loader.load()
    return pages

def text_splitters(data):
    text_splitter = CharacterTextSplitter(
                                    separator='\n', 
                                    chunk_size=1000, 
                                    chunk_overlap=200
                                )
    docs = text_splitter.split_documents(data)
    return docs

def create_llm(docs, local_vector_dir: Path):
    embeddings = OpenAIEmbeddings(
        openai_api_base=API_BASE,
        openai_api_type=API_TYPE,
    )
    file_name = "faiss_index"
    path_to_vectordb = local_vector_dir.joinpath(file_name)
    if len(os.listdir(local_vector_dir)) == 0:
        vectorStore_openAI = FAISS.from_documents(docs, embeddings)
        vectorStore_openAI.save_local(path_to_vectordb)
    else:
        vectorStore_openAI = FAISS.load_local(path_to_vectordb, embeddings)

    llm = AzureChatOpenAI(deployment_name="WBU-GPT-4",
                      model_name=openai.api_type,
                      openai_api_base=openai.api_base,
                      openai_api_version=openai.api_version,
                      openai_api_key= openai.api_key,
                      temperature = 0.0)

    return llm, vectorStore_openAI

def create_chain(llm, vector_store):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain

if __name__ == "__main__":
    
    pdf_dataset_directory =  r'./PDFBased/KnowledgeBase_PDF/'
    local_save_directory = Path('./PDFBased/pdf_vector_store/')
    
    data = load_pdfs(pdf_dataset_directory)
    docs = text_splitters(data)
    llm, vectorStore_openAI = create_llm(docs, local_save_directory)

    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vectorStore_openAI.as_retriever(),
                return_source_documents=True
            )

    question2 = "What are the health impacts of cannabis legalization in Canada?"

    result = qa_chain({"query": question2})
    print('Answer: ', result['result'])
    print('Source: ', result["source_documents"][0])