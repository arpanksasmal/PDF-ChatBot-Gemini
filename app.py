import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Google API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not found in environment variables.")
    st.error("API key is missing. Please set up the API key in your .env file.")
    st.stop()

genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        logger.error(f"Error extracting text from PDFs: {e}")
        st.error("An error occurred while processing the PDF files.")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        logger.error(f"Error creating or saving vector store: {e}")
        st.error("An error occurred while creating the vector store.")

def get_conversational_chain(model_choice, temperature):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model=model_choice, temperature=temperature)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        logger.error(f"Error initializing conversational chain: {e}")
        st.error("An error occurred while setting up the conversational chain.")
        return None

def user_input(user_question, model_choice, temperature):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain(model_choice, temperature)
        if chain:
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            st.write("Reply: ", response.get("output_text", "No response"))
        else:
            st.write("Error in chain setup.")
    except Exception as e:
        logger.error(f"Error in user input processing: {e}")
        st.error("An error occurred while processing your request.")

def main():
    st.set_page_config(page_title="Chat with Multiple PDF")
    st.header("Chat with Multiple PDF using Gemini")

    # Add simplified model names and map them to full model paths
    model_mapping = {
        "gemini-1.0-pro-latest": "models/gemini-1.0-pro-latest",
        "gemini-1.0-pro": "models/gemini-1.0-pro",
        "gemini-pro": "models/gemini-pro",
        "gemini-1.0-pro-001": "models/gemini-1.0-pro-001",
        "gemini-1.5-pro-latest": "models/gemini-1.5-pro-latest",
        "gemini-1.5-pro-001": "models/gemini-1.5-pro-001",
        "gemini-1.5-pro": "models/gemini-1.5-pro",
        "gemini-1.5-pro-exp-0801": "models/gemini-1.5-pro-exp-0801",
        "gemini-1.5-pro-exp-0827": "models/gemini-1.5-pro-exp-0827",
        "gemini-1.5-flash-latest": "models/gemini-1.5-flash-latest",
        "gemini-1.5-flash-001": "models/gemini-1.5-flash-001",
        "gemini-1.5-flash-001-tuning": "models/gemini-1.5-flash-001-tuning",
        "gemini-1.5-flash": "models/gemini-1.5-flash",
        "gemini-1.5-flash-exp-0827": "models/gemini-1.5-flash-exp-0827",
        "gemini-1.5-flash-8b-exp-0827": "models/gemini-1.5-flash-8b-exp-0827"
    }
    model_names = list(model_mapping.keys())

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and click on Submit & Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")
                    else:
                        st.error("No text extracted from PDFs.")
                else:
                    st.error("Please upload PDF files.")

        model_choice = st.selectbox("Choose a Gemini Model", model_names)
        temperature = st.slider("Set Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

    selected_model_path = model_mapping[model_choice]

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, selected_model_path, temperature)

if __name__ == "__main__":
    main()
