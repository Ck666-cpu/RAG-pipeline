from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import qdrant_client


# --- GLOBAL SETTINGS ---
# We tell LlamaIndex to ALWAYS use FastEmbed for embeddings (instead of OpenAI)
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")


def get_index(collection_name="crag_llamaindex"):
    """
    Connects to Qdrant and returns the Searchable Index.
    """
    # 1. Connect to Qdrant Client
    client = qdrant_client.QdrantClient(url="http://localhost:6333")

    # 2. Connect LlamaIndex to Qdrant
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

    # 3. Create a 'Storage Context' (The bridge between LlamaIndex and Qdrant)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Load the Index
    # We use from_vector_store to load existing data without re-creating it
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )
    return index


def upload_data(text_list, collection_name="crag_llamaindex"):
    """
    Takes raw text, converts to Documents, and saves to Qdrant.
    """
    print("Initializing Database...")

    # Connect to Qdrant
    client = qdrant_client.QdrantClient(url="http://localhost:6333")
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Convert Strings -> LlamaIndex Documents
    documents = [Document(text=t) for t in text_list]

    # Create the Index (This automatically embeds and upserts!)
    # We throw away the return value because we just want to save it.
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    print(f"Successfully uploaded {len(text_list)} documents to '{collection_name}'!")

def upload_file(file_path, collection_name="crag_llamaindex"):
    """ (NEW) Reads a file (PDF/TXT) and uploads it """
    print(f"Reading file: {file_path}...")

    # 1. Read the file (LlamaIndex handles the format automatically)
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()

    print(f"   -> Found {len(documents)} pages/sections. Uploading to Qdrant...")

    # 2. Connect and Upload
    client = qdrant_client.QdrantClient(url="http://localhost:6333")
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3. Create Index (Embed & Save)
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print("   -> Success! File indexed.")