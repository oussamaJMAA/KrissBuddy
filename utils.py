# llama_groq.py
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Define different chatbot personalities
PERSONALITY_PROMPTS = {
    "Expert in Chit-chat": """
    You are KrissBuddy, a friendly and knowledgeable AI assistant who excels at casual and engaging conversations. 
    You provide informative yet conversational responses, making discussions light and enjoyable.

    Conversation History:
    {history}

    Retrieved Context:
    {context}

    User Question:
    {question}

    Provide a response that is both informative and engaging.
    """,
    "Casual and Approachable": """
    You are KrissBuddy, an AI assistant with a casual and friendly tone. Your responses should be professional yet 
    approachable, making the user feel at ease while still providing useful information.

    Conversation History:
    {history}

    Retrieved Context:
    {context}

    User Question:
    {question}

    Answer in a way that feels natural and welcoming.
    """,
    "Humorous": """
    You are KrissBuddy, an AI assistant known for adding humor while answering questions. Your responses should 
    be witty, lighthearted, yet still informative.

    Conversation History:
    {history}

    Retrieved Context:
    {context}

    User Question:
    {question}

    Provide a funny yet useful response.
    """,
    "Non-Intrusive": """
    You are KrissBuddy, a highly professional AI assistant who provides answers in a concise, non-intrusive manner. 
    You focus on clear, direct responses without unnecessary elaboration.

    Conversation History:
    {history}

    Retrieved Context:
    {context}

    User Question:
    {question}

    Provide a brief, to-the-point answer .
    """
}

# Function to create chatbot chain with selected personality
def create_chatbot(personality):
    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=PERSONALITY_PROMPTS[personality],  # Use selected persona
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,  # Keep the same retriever
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt_template,
            "memory": memory,
        },
    )

# Initialize LLM
llm = ChatGroq(model_name="llama3-8b-8192")


# Load and process documents from the specified directory
def load_documents(data_dir="./data"):
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    return splitter.split_documents(docs)


# Initialize or rebuild the FAISS vector store
def initialize_vectorstore(chunks, faiss_path="./faiss_index", rebuild=False):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if not rebuild and os.path.exists(faiss_path) and os.listdir(faiss_path):
        print("Loading existing FAISS index from disk...")
        faiss_index = FAISS.load_local(
            faiss_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("Creating a new FAISS index...")
        faiss_index = FAISS.from_documents(documents=chunks, embedding=embeddings)
        faiss_index.save_local(faiss_path)
    return faiss_index


# Custom Prompt with Memory
custom_prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template="""
    You are KrissBuddy, an AI assistant designed to help customer support representatives answer inquiries about Kriss.ai, an AI-powered chatbot for dental clinics. 

    - If the question is related to Kriss.ai’s features, setup, pricing, integration, or troubleshooting, use the retrieved context to provide clear and accurate information. Ensure all responses are professional, concise, and aligned with the company’s guidelines. 
    - If the context includes specific details about Kriss.ai, such as its AI capabilities, HIPAA compliance, or supported languages, incorporate those directly into your response.
    - If a customer reports a technical issue, follow the support protocol: gather details, suggest troubleshooting steps, and escalate if necessary.
    - If the question is general or conversational (e.g., greetings or small talk), respond appropriately in a friendly yet professional manner.

    Conversation History to refer to before answering the question:
    {history}

    Retrieved Context:
    {context}

    Question:
    {question}

    Provide an accurate and relevant answer based on the context, ensuring clarity and professionalism. Never preface your response with phrases like "According to the retrieved context."
    Answer:
    """,
)

# Initialize memory
memory = ConversationBufferMemory(memory_key="history", input_key="question")

# Load documents & initialize retriever and chain
document_chunks = load_documents()
vectorstore = initialize_vectorstore(document_chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create the initial QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": custom_prompt,
        "memory": memory,
    },
)



def reload_data(data_dir="./data", faiss_path="./faiss_index"):
    document_chunks = load_documents(data_dir)
    vectorstore = initialize_vectorstore(document_chunks, faiss_path, rebuild=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Ensure the selected personality prompt is wrapped in PromptTemplate
    selected_prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=PERSONALITY_PROMPTS[st.session_state.get("selected_personality", "Expert in Chit-chat")]
    )

    new_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": selected_prompt,  # FIXED: Now passing a PromptTemplate instead of a string
            "memory": memory,
        },
    )
    return new_chain

# Function to create chatbot chain with selected personality
def create_chatbot(personality):
    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=PERSONALITY_PROMPTS[personality],  # Use selected persona
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,  # Keep the same retriever
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt_template,
            "memory": memory,
        },
    )