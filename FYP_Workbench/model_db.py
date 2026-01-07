# FYP_Workbench/model_db.py
import os
import qdrant_client
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterCondition

# CONFIGURATION
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
QDRANT_URL = "http://127.0.0.1:6333"


def get_client():
    try:
        return qdrant_client.QdrantClient(url=QDRANT_URL)
    except Exception:
        return None


def get_index(collection_name):
    client = get_client()
    if not client: return None
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


# --- UPDATED UPLOAD FUNCTION ---
def upload_file(file_path, collection_name, owner_username, visibility="private"):
    """
    visibility: 'private' (owner only) or 'global' (everyone)
    """
    if not os.path.exists(file_path):
        return False, "File path not found."

    client = get_client()
    if not client: return False, "Qdrant is offline."

    try:
        # Load documents
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()

        # INJECT METADATA (Permissions)
        for doc in documents:
            doc.metadata["file_name"] = os.path.basename(file_path)
            doc.metadata["owner"] = owner_username
            doc.metadata["visibility"] = visibility

            # Indexing
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        return True, "Upload Successful"
    except Exception as e:
        return False, str(e)


# --- NEW: PERMISSION FILTER GENERATOR ---
def get_user_filters(username):
    """
    Creates a Qdrant Filter: (visibility == 'global') OR (owner == username)
    """
    return MetadataFilters(
        filters=[
            MetadataFilter(key="visibility", value="global"),
            MetadataFilter(key="owner", value=username),
        ],
        condition=FilterCondition.OR  # Match EITHER condition
    )