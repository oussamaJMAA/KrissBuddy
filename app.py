# app.py

import streamlit as st
import datetime
import os
from utils import qa_chain, reload_data, create_chatbot

# Start with the initially imported chain
if "chatbot" not in st.session_state:
    st.session_state.chatbot = qa_chain

st.set_page_config(page_title="KrissBuddy")

# CSS Styling
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        button:hover {
            border: 2px solid #87CEEB !important;
            color: #87CEEB !important;
            background-color: transparent !important;
        }
        div[data-testid="stSidebarNav"] > div > div:hover {
            border: 2px solid #87CEEB !important;
        }
        div[data-baseweb="select"]:hover,
        div[data-testid="stTextInput"] > div:hover {
            border: 2px solid #87CEEB !important;
        }

    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("📁 Document Manager")

    st.markdown("### 📄 Document Upload")
    st.caption(
        "Upload PDF documents to enhance chat responses with relevant information"
    )

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()  # Store processed filenames

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="uploaded_files",
        label_visibility="collapsed",
    )

    # Clear processed files if needed
    if st.button("Clear Uploads"):
        st.session_state.processed_files.clear()
        st.success("Uploads cleared!")

    # Process new files only if they haven't been processed
    new_files = [
        f for f in uploaded_files if f.name not in st.session_state.processed_files
    ]

    if new_files:
        with st.spinner("Processing uploaded files..."):
            try:
                data_dir = "./data"
                os.makedirs(data_dir, exist_ok=True)

                for uploaded_file in new_files:
                    file_path = os.path.join(data_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Mark file as processed
                    st.session_state.processed_files.add(uploaded_file.name)

                # Reload chatbot with new data
                st.session_state.chatbot = reload_data()
                st.success("Files processed and data reloaded successfully!")

            except Exception as e:
                st.error(f"Error processing uploaded files: {e}")

    st.divider()

    st.markdown("### 🦸 Chatbot Persona")

    if "selected_personality" not in st.session_state:
        st.session_state.selected_personality = "Expert in Chit-chat"

    persona = st.selectbox(
        "Select Assistant Personality:",
        ("Expert in Chit-chat", "Casual and Approachable", "Humorous", "Non-Intrusive"),
        index=[
            "Expert in Chit-chat",
            "Casual and Approachable",
            "Humorous",
            "Non-Intrusive",
        ].index(st.session_state.selected_personality),
    )

    # If personality changes, update prompt dynamically (NO FAISS RELOAD)
    if persona != st.session_state.selected_personality:
        st.session_state.selected_personality = persona
        st.session_state.chatbot = create_chatbot(persona)  # Only update prompt
        st.rerun()  # Refresh UI

    st.divider()

    st.markdown(
        """
    **Tips:**
    - Upload documents first for best results
    - Switch personas anytime
    - Files are kept private
    """
    )

# Main App UI
if not st.session_state.messages:
    current_hour = datetime.datetime.now().hour
    greeting = (
        "Good Morning 🌞"
        if 5 <= current_hour < 12
        else "Good Afternoon 🌇" if 12 <= current_hour < 18 else "Good Night 🌙"
    )

    st.markdown(
        f"""
        <h1 style='text-align: center;'>Welcome to KrissBuddy 😊</h1>
        <div style="text-align: center;">
            <h3 style="color: #87CEEB;">{greeting}</h3>
            <p style="color: #64748b; font-size: 16px;">
                I am your personal intelligent assistant KrissBuddy.<br>
                How can I assist you today?
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    questions = [
        "How do I explain Kriss.ai's setup fee and monthly pricing in a DM?",
        "What's the best way to describe Kriss.ai's key features in a short message?",
        "How do I clearly explain how Kriss.ai reduces no-shows and cancellations?",
        "They're asking if Kriss.ai can handle multiple languages. What's the best reply?",
        "How does Kriss.ai ensure patient data security and sensitive information?",
        "What's the onboarding process for new clients trying to learn about Kriss.ai?",
    ]

    for i in range(0, len(questions), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(questions):
                with cols[j]:
                    question = questions[i + j]
                    if st.button(question, use_container_width=True, key=f"q_{i+j}"):
                        # Append user message only once
                        st.session_state.messages.append(
                            {"role": "user", "content": question}
                        )

                        with st.spinner("Thinking... 🤔"):
                            response = st.session_state.chatbot.invoke(
                                {"query": question}
                            )["result"]

                        formatted_response = response.replace(
                            "\n", "  \n"
                        )  # Ensure proper formatting
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"**Response:**\n\n{formatted_response}",
                            }
                        )

                        st.rerun()


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Chat Input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking... 🤔"):
        response = st.session_state.chatbot.invoke({"query": prompt})["result"]

    formatted_response = response.replace("\n", "  \n")  # Ensure proper formatting
    print("formatted_response", formatted_response)
    st.session_state.messages.append(
        {"role": "assistant", "content": f"**Response:**\n\n{formatted_response}"}
    )

    st.rerun()
