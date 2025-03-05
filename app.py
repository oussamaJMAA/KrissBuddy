# app.py

import streamlit as st
import datetime  # For time detection
import os
from utils import qa_chain, reload_data

# Start with the initially imported chain
chatbot = qa_chain

st.set_page_config(page_title="KrissBuddy")

custom_css = """
    <style>
        /* Buttons (Primary & Secondary) */
        button:hover {
            border: 2px solid #87CEEB !important;  /* Light Blue Border */
            color: #87CEEB !important;  /* Change Text Color */
            background-color: transparent !important;  /* Keep Background Transparent */
        }

        /* Sidebar Nav Items */
        div[data-testid="stSidebarNav"] > div > div:hover {
            border: 2px solid #87CEEB !important;  /* Light Blue Border */
            color: #87CEEB !important;  /* Change Text Color */
            background-color: transparent !important;  /* No Background Fill */
        }

        /* Selectbox, Text Input, etc. */
        div[data-baseweb="select"]:hover,
        div[data-testid="stTextInput"] > div:hover {
            border: 2px solid #87CEEB !important;
            color: #87CEEB !important;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown(
    """
<style>
    .chat-history {
        margin-bottom: 200px;
    }
    .welcome-section {
        margin: 20px 0 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
# Sidebar
with st.sidebar:
    st.title("📁 Document Manager")

    st.markdown("### 📄 Document Upload")
    st.caption("Upload PDF documents to enhance chat responses with relevant information")
    
    # Initialize a session flag to track processing
    if "files_uploaded_processed" not in st.session_state:
        st.session_state["files_uploaded_processed"] = False

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="uploaded_files",  # assign a key to the uploader
        label_visibility="collapsed",
    )
    
    # Button to clear the processed flag (allowing new uploads to be processed)
    if st.button("Clear Uploads"):
        st.session_state["files_uploaded_processed"] = False

    # Process files only if they haven't been processed already
    if uploaded_files and not st.session_state["files_uploaded_processed"]:
        with st.spinner("Processing uploaded files..."):
            try:
                data_dir = "./data"
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(data_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                # Reload the data to update the vectorstore and QA chain
                chatbot = reload_data()
                st.success("Files processed and data reloaded successfully!")
                # Set flag to prevent reprocessing on rerun
                st.session_state["files_uploaded_processed"] = True
            except Exception as e:
                st.error(f"Error processing uploaded files: {e}")
            finally:
                st.rerun()

        

    st.divider()

    # Persona Selection
    st.markdown("### 🦸 Chatbot Persona")
    persona = st.selectbox(
        "Select Assistant Personality:",
        ("General Assistant", "Medical Expert", "Technical Support", "Sales Advisor"),
        label_visibility="collapsed",
    )

    st.divider()

    # Help Text
    st.markdown(
        """
    **Tips:**
    - Upload documents first for best results
    - Switch personas anytime
    - Files are kept private
    """
    )

# Main app
if not st.session_state.messages:
    # Time-based greeting
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "Good Morning 🌞"
    elif 12 <= current_hour < 18:
        greeting = "Good Afternoon 🌇"
    else:
        greeting = "Good Night 🌙"

    st.markdown(
        """
    <h1 style='
        text-align: center;
        margin: -20px 0 10px 0;
        padding: 0;
        font-size: 2.5rem;
    '>
        Welcome to KrissBuddy 😊
    </h1>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
    <div class="welcome-section" style="text-align: center;">
        <h3 style="color: #87CEEB; margin-bottom: 16px;">{greeting}</h3>
        <p style="color: #64748b; font-size: 16px;">
            I am your personal intelligent assistant KrissBuddy<br>
            How can I assist you today?
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Recommended questions 
    questions = [
        "How do I explain Kriss.ai's setup fee and monthly pricing in a DM?",
        "What's the best way to describe Kriss.ai's key features in a short message?",
        "How do I clearly explain how Kriss.ai reduces no-shows and cancellations?",
        "They're asking if Kriss.ai can handle multiple languages. What's the best reply?",
        "How does Kriss.ai ensure patient data security and sensitive information? ",
        "What's the onboarding process for new clients trying to learn about Kriss.ai?",
    ]

    for i in range(0, len(questions), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(questions):
                with cols[j]:
                    question = questions[i + j]
                    if st.button(
                        question,
                        use_container_width=True,
                        key=f"q_{i+j}",
                        help="Click to ask",
                    ):
                        st.session_state.messages.extend(
                            [
                                {"role": "user", "content": question},
                                {
                                    "role": "assistant",
                                    "content": f"**Response:**\n\n{chatbot.invoke({'query': question})['result']}",
                                },
                            ]
                        )
                        st.rerun()

# Display chat history
with st.container():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.extend(
        [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": f"**Response:**\n\n{chatbot.invoke({'query': prompt})['result']}",
            },
        ]
    )
    st.rerun()
