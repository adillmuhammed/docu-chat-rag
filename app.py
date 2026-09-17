import os

import gradio as gr
from dotenv import load_dotenv

from chunks import create_database

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model


# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# Open Chroma database
vector_store = Chroma(
    collection_name="Research_collections",
    embedding_function=embedding_model,
    persist_directory="./chroma_langchain_db"
)


# Groq model
model = init_chat_model(
    "groq:allam-2-7b",
    api_key=GROQ_API_KEY,
)


# Retrieve relevant chunks
def retrieve_context(query):

    retrieved_docs = vector_store.similarity_search(
        query,
        k=4
    )

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    return context


# Ask question
def ask_about_pdf(user_query):

    if not user_query.strip():
        return "Please enter a question."

    context = retrieve_context(user_query)

    system_message = f"""
        You are a helpful PDF research assistant.

        Answer the user's question using ONLY the context below.

        Rules:
        - Give only the answer.
        - Do not mention chunks.
        - Do not mention metadata.
        - Do not mention ChromaDB.
        - Do not mention retrieval.
        - Do not make up information.
        - If the answer is not available in the context,
        say: "I couldn't find the answer in the provided PDF."

        Context:
        {context}
        """

    response = model.invoke([
        ("system", system_message),
        ("human", user_query)
    ])

    return response.content

with gr.Blocks(title="PDF Research Assistant") as app:

    gr.Markdown(
        """
        # 📚 PDF Research Assistant

        Upload a PDF, add it to the knowledge base,
        and ask questions about it.
        """
    )

    # PDF section
    gr.Markdown("## 📄 1. Add a PDF")

    pdf_file = gr.File(
        label="Upload PDF",
        file_types=[".pdf"],
        type="filepath"
    )

    add_pdf_button = gr.Button(
        "➕ Add PDF to Knowledge Base"
    )

    upload_status = gr.Textbox(
        label="Status",
        interactive=False
    )

    add_pdf_button.click(
        fn=create_database,
        inputs=pdf_file,
        outputs=upload_status
    )

    # Question section
    gr.Markdown("## 💬 2. Ask a Question")

    question = gr.Textbox(
        label="Your Question",
        placeholder="Ask something about the uploaded PDF..."
    )

    ask_button = gr.Button(
        "🔍 Ask"
    )

    answer = gr.Markdown(
        label="Answer"
    )

    ask_button.click(
        fn=ask_about_pdf,
        inputs=question,
        outputs=answer
    )


app.launch()