import os
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

# --- GLOBAL SETTINGS ---
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
QDRANT_URL = "http://localhost:6333"


def get_client():
    """
    Safely connects to Qdrant. Returns None if connection fails.
    """
    try:
        client = qdrant_client.QdrantClient(url=QDRANT_URL)
        # Light check to see if server is responsive
        client.get_collections()
        return client
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not connect to Qdrant at {QDRANT_URL}.")
        print(f"   Details: {e}")
        print("   -> Is Docker running? (docker start my_qdrant)")
        return None


def get_index(collection_name="crag_llamaindex"):
    client = get_client()
    if not client:
        raise ConnectionError("Qdrant Client is offline.")

    try:
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    except Exception as e:
        print(f"⚠️ Warning: Could not load index for '{collection_name}'. (It might be empty).")
        return None


def upload_file(file_path, collection_name="crag_llamaindex"):
    # 1. VALIDATION: Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at path: {file_path}")
        return

    print(f"Processing file: {file_path}...")

    try:
        # 2. VALIDATION: Check if file is readable
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()

        if not documents:
            print("⚠️ Warning: File was read but contained no text.")
            return

        client = get_client()
        if not client: return

        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        print("   -> Success! Document stored.")

    except Exception as e:
        print(f"❌ Error during upload: {e}")


def list_uploaded_files(collection_name="crag_llamaindex"):
    client = get_client()
    if not client: return []

    try:
        # Check if collection exists first to avoid crashes
        if not client.collection_exists(collection_name):
            return []

        scroll_result, _ = client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )

        unique_files = set()
        for point in scroll_result:
            if point.payload and 'file_name' in point.payload:
                unique_files.add(point.payload['file_name'])

        return list(unique_files)

    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return []


def delete_file(file_name, collection_name="crag_llamaindex"):
    client = get_client()
    if not client: return

    print(f"Attempting to delete: {file_name}...")

    try:
        file_filter = Filter(
            must=[FieldCondition(key="file_name", match=MatchValue(value=file_name))]
        )

        client.delete(
            collection_name=collection_name,
            points_selector=file_filter
        )
        print(f"   -> Successfully removed '{file_name}'.")

    except Exception as e:
        print(f"❌ Error deleting file: {e}")