# # Import libraries
import ast
import json
import os

import pypdf
from streamlit.runtime.state import SessionState
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from sympy import pretty_print

st = SessionState()

def count_symbols_in_pdf(pdf_path: str) -> int:
    num_of_words = 0
    pdf = pypdf.PdfReader(pdf_path)
    for page in pdf.pages:
        text = page.extract_text()
        text = ''.join(e for e in text if e.isalnum())
        num_of_words += len(text)
    return num_of_words, len(pdf.pages)

# # Init ollama client

model = "mixtral:8x7b"
llm = Ollama(
    base_url="http://localhost:11434", model=model, verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

def process_article(pdf_name):
    pdf_path = f"data/articles/{pdf_name}"
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10_000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)

    # # Save data to the vector database

    persist_directory = f"data/vectorstore/{pdf_name}"
    embedding_model = OllamaEmbeddings(base_url='http://localhost:11434', model=model)
    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=embedding_model, persist_directory=persist_directory
    )
    vectorstore.persist()


    retriever = vectorstore.as_retriever()

    # # Prompt template

    template = """
    Context: {context}
    History: {history}
    
    User: {question}
    
    User can provide you a scientific article. Your task is to create a LONG (around a 1 A4 page long) summary of it. In this summary you must provide the next information:
    - the problem researchers tried to solve, it's background and existing solutions (if any)
    - the solution researchers have provided, main components, principles and techniques used
    - results of the solution, how it is comparable with other solutions.
    Hints:
    - don't copy used formulas, just describe them using words
    - don't delve into other researchers' solutions, unless needed
    - try to keep the background short. Very often researchers will provide a whole section to describe all needed knowledge around the problem. Keep it simple.

    !!!!!!! MANDATORY !!!!!!!!
        Mandatory:
    - don't copy the abstract
    - don't split summary into the sections
    - don't copy used formulas, just describe them using words
    - when you use abbreviations for the first time - use the full name as well to describe it
    }}
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )

    # # Init session state

    st.template = template
    st.prompt = prompt
    st.memory = memory
    st.vectorstore = vectorstore
    st.vectorstore.persist()
    st.retriever = st.vectorstore.as_retriever()
    st.llm = llm
    st.chat_history = []

    # # Creating a Q&A chain

    st.qa_chain = RetrievalQA.from_chain_type(
        llm=st.llm,
        chain_type='stuff',
        retriever=st.retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": st.prompt,
            "memory": st.memory,
        }
    )
    response = st.qa_chain("Make a summary of the article")["result"]
    with open(f"data/summaries/{pdf_name}.json", "w", encoding='utf-8') as f:
        response = {"summary": response}
        response["character_count"], response["pages"] = count_symbols_in_pdf(pdf_path)
        response["pdf_name"] = pdf_name
        json.dump(response, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    for pdf_name in os.listdir("data/articles"):
        process_article(pdf_name)