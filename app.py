from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import transformers 
from pdf2image import convert_from_path
from transformers import AutoTokenizer, TextStreamer, pipeline
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModel
import pypdf
import tensorflow as tf
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR
import pydantic
from pydantic import BaseModel, field_validator, model_validator

from langchain_community.vectorstores import DeepLake
from langchain_text_splitters import CharacterTextSplitter


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set the path for poppler
poppler_path = r'C:\Users\R Vignesh\Documents\poppler-24.07.0\Library\bin'

# Convert PDFs to images
meta_images = convert_from_path("pdfs/Meta-03-31-2024-Exhibit-99-1_FINAL.pdf", poppler_path=poppler_path, dpi=88)
nvidia_images = convert_from_path("pdfs/NVIDIAAn.pdf", poppler_path=poppler_path, dpi=88)
tesla_images = convert_from_path("pdfs/tsla-20240102-gen.pdf", poppler_path=poppler_path, dpi=88)

# Load PDF documents
loader = PyPDFDirectoryLoader("pdfs")
docs = loader.load()

embeddings= HuggingFaceInstructEmbeddings(model_name = "BAAI/llm-embedder",
model_kwargs = {'device': 'cpu'}, encode_kwargs = {'normalize_embeddings': True})

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(docs)

db = DeepLake(dataset_path="./my_deeplake/", embedding=embeddings, overwrite=True)
db.add_documents(docs)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

# Define system prompt
DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()

# Define prompt generation function
def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{prompt} [/INST]
""".strip()

# Initialize text generation pipeline
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,  # Enable sampling mode
    temperature=0.7,  # Adjust temperature to control randomness
    top_p=0.95,  # Use nucleus sampling
    repetition_penalty=1.15,
    streamer=streamer,
)

# Create a HuggingFace pipeline
llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.7})  # Ensure temperature matches sampling

# Define system prompt for QA chain
SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

# Generate the template for the prompt
template = generate_prompt(
    """
{context}

Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Initialize the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

# Function to get answer from the QA chain
def get_answer(question: str):
    return qa_chain.invoke(question)

# Streamlit UI for Chatbot
st.title("AI Chatbot with LangChain and HuggingFace")
st.write("This chatbot can answer questions about the provided PDF documents.")

# User input
user_input = st.text_input("Ask a question:")

# Generate and display answer
if user_input:
    with st.spinner("Generating response..."):
        answer = get_answer(user_input)
        st.write("Answer:")
        st.write(answer['result'])

        st.write("Source Documents:")
        for doc in answer['source_documents']:
            st.write(doc.metadata['source'], doc.page_content)