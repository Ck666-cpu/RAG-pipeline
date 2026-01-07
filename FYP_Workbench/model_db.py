# FYP_Workbench/model_db.py
import os
import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- CONFIGURATION ---
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
QDRANT_URL = "http://127.0.0.1:6333"


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


# Add this to FYP_Workbench/model_db.py if needed
def list_uploaded_files(collection_name):
    client = get_client()
    if not client: return []
    # Note: Qdrant doesn't store filenames natively unless in metadata.
    # This requires scrolling through points to find unique filenames.
    # For a simple prototype, we might skip this or implement a scroll.
    return []


def delete_file(file_name, collection_name):
    client = get_client()
    if not client: return

    # Qdrant delete by filter
    client.delete(
        collection_name=collection_name,
        points_selector=qdrant_client.models.FilterSelector(
            filter=qdrant_client.models.Filter(
                must=[
                    qdrant_client.models.FieldCondition(
                        key="file_name",
                        match=qdrant_client.models.MatchValue(value=file_name)
                    )
                ]
            )
        )
    )