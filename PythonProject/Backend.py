import model_db
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import PromptTemplate

# --- CONFIGURATION ---
COLLECTION_NAME = "crag_llamaindex"

qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query in a clear, step-by-step format.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

# 1. SETUP LLM
llm = Ollama(
    model="phi3",
    request_timeout=360.0,
    context_window=2048,
    additional_kwargs={"num_ctx": 2048}
)
Settings.llm = llm

# 2. SETUP RERANKER
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=1
)

# 3. SETUP DATABASE
try:
    index = model_db.get_index(COLLECTION_NAME)
except:
    print("Index not found. Please upload a document first.")
    index = None


# --- NEW FUNCTION: INPUT RECEIVER ---
def receive_user_query():
    """
    This function simulates the 'Frontend'.
    It waits for the user to type a question.
    """
    print("\n" + "-" * 30)
    query = input("❓ Enter your query: ").strip()
    return query


# --- CRAG LOGIC (Now returns the answer) ---
def run_crag(user_query):
    if not index:
        return "System Error: Please upload a document first."

    print(f"⚙️ Processing: '{user_query}'")

    # 1. RETRIEVE
    retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
    retrieved_nodes = retriever.retrieve(user_query)

    if not retrieved_nodes:
        print("🔴 CRAG: No documents found.")
        return fallback_to_general_knowledge(user_query)

    # 2. RERANK
    reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_str=user_query)

    if not reranked_nodes:
        return fallback_to_general_knowledge(user_query)

    # 3. EVALUATE
    best_score = reranked_nodes[0].score
    print(f"   -> Relevance Score: {best_score:.4f}")

    if best_score < 0.5:
        print("🔴 CRAG DECISION: Database Irrelevant.")
        return fallback_to_general_knowledge(user_query)

    # 4. GENERATE
    print("🟢 CRAG DECISION: Database Good.")
    synthesizer = get_response_synthesizer(response_mode="tree_summarize", text_qa_template=qa_prompt_tmpl)
    response = synthesizer.synthesize(user_query, nodes=reranked_nodes)

    # RETURN the answer (don't just print it)
    return str(response)


def fallback_to_general_knowledge(query):
    print(f"🧠 Switching to General Knowledge...")
    response = llm.complete(query)
    return str(response)


# --- UPLOAD FUNCTION ---
def upload_document_console():
    print("\n📂 --- FILE UPLOAD ---")
    file_path = input("Paste file path: ").strip().replace('"', '')
    if file_path:
        model_db.upload_file(file_path, COLLECTION_NAME)
        global index
        index = model_db.get_index(COLLECTION_NAME)
    else:
        print("Cancelled.")


# --- MAIN CONTROLLER ---
if __name__ == "__main__":
    while True:
        print("\n=== BACKEND MENU ===")
        print("1. Start Query Session")
        print("2. Upload Document")
        print("3. Exit")

        choice = input("Select: ")

        if choice == "1":
            # 1. GET INPUT
            user_text = receive_user_query()

            if user_text:
                # 2. PROCESS INPUT
                final_answer = run_crag(user_text)

                # 3. DISPLAY OUTPUT (Simulating Frontend display)
                print(f"\n🤖 FINAL ANSWER:\n{final_answer}")

        elif choice == "2":
            upload_document_console()
        elif choice == "3":
            break