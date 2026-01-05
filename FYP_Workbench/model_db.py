# FYP_Workbench/model_db.py
import os
import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# --- CONFIGURATION ---
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
QDRANT_URL = "http://localhost:6333"


def get_client():
    """Safely connects to Qdrant."""
    try:
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        client.get_collections()  # Test connection
        return client
    except Exception as e:
        print(f"❌ [model_db] Qdrant Connection Error: {e}")
        print("   -> Make sure you are running Qdrant in Docker!")
        return None


def get_index(collection_name):
    client = get_client()
    if not client: return None

    try:
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    except Exception:
        return None


def upload_file(file_path, collection_name):
    if not os.path.exists(file_path):
        return False, "File path not found."

    client = get_client()
    if not client: return False, "Qdrant is offline."

    try:
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()

        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        return True, "Upload Successful"
    except Exception as e:
        return False, str(e)