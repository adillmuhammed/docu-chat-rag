import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# Chroma database
vector_store = Chroma(
    collection_name="Research_collections",
    embedding_function=embedding_model,
    persist_directory="./chroma_langchain_db"
)


def create_database(file):

    if file is None:
        return "❌ Please upload a PDF."

    try:

        # Load PDF
        loader = PyPDFLoader(file)
        docs = loader.load()

        # Split PDF
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )

        all_splits = text_splitter.split_documents(docs)

        # Existing IDs
        existing_ids = set(
            vector_store.get()["ids"]
        )

        new_documents = []
        new_ids = []

        # Create IDs
        for doc in all_splits:

            chunk_text = doc.page_content

            chunk_id = hashlib.sha256(
                chunk_text.encode("utf-8")
            ).hexdigest()

            if chunk_id not in existing_ids:

                new_documents.append(doc)
                new_ids.append(chunk_id)

        # Add new chunks
        if new_documents:

            vector_store.add_documents(
                documents=new_documents,
                ids=new_ids
            )

            return (
                f"✅ PDF processed successfully!\n\n"
                f"📄 Total chunks: {len(all_splits)}\n"
                f"➕ New chunks added: {len(new_documents)}\n"
                f"⏭️ Duplicate chunks skipped: "
                f"{len(all_splits) - len(new_documents)}"
            )

        return (
            f"✅ PDF already exists in the knowledge base.\n\n"
            f"📄 Total chunks: {len(all_splits)}\n"
            f"⏭️ Duplicate chunks skipped: {len(all_splits)}"
        )

    except Exception as e:

        return f"❌ Error: {str(e)}"