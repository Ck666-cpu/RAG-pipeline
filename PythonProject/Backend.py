import sys
import model_db
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, get_response_synthesizer, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank

# --- CONFIGURATION ---
COLLECTION_NAME = "crag_llamaindex"

# 1. SETUP LLM
try:
    llm = Ollama(
        model="phi3",
        request_timeout=360.0,
        context_window=2048,
        additional_kwargs={"num_ctx": 2048}
    )
    Settings.llm = llm
except Exception as e:
    print(f"❌ Error initializing Ollama: {e}")
    print("   -> Is Ollama running? (ollama serve)")
    sys.exit(1)  # Stop program if LLM is dead

# 2. SETUP PROMPT
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

# 3. SETUP RERANKER
print("Initializing AI Models...")
try:
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=1
    )
except Exception as e:
    print(f"❌ Error downloading Reranker model: {e}")
    print("   -> Check your internet connection.")
    sys.exit(1)

# 4. SETUP DATABASE
index = None
try:
    index = model_db.get_index(COLLECTION_NAME)
except Exception as e:
    print(f"⚠️ Initial Database Connection Failed: {e}")


# --- INPUT RECEIVER ---
def receive_user_query():
    print("\n" + "-" * 30)
    return input("❓ Enter your query: ").strip()


# --- CRAG LOGIC ---
def run_crag(user_query):
    # Validation: Check if index exists
    if not index:
        return "System Error: Database index is not loaded. Please upload a document."

    print(f"⚙️ Processing: '{user_query}'")

    try:
        # 1. RETRIEVE
        retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
        retrieved_nodes = retriever.retrieve(user_query)

        if not retrieved_nodes:
            return fallback_to_general_knowledge(user_query)

        # 2. RERANK
        reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_str=user_query)

        if not reranked_nodes:
            return fallback_to_general_knowledge(user_query)

        # 3. EVALUATE
        best_score = reranked_nodes[0].score
        print(f"   -> Relevance Score: {best_score:.4f}")

        if best_score < 0.2:
            print("🔴 CRAG DECISION: Database Irrelevant.")
            return fallback_to_general_knowledge(user_query)

        # 4. GENERATE
        print("🟢 CRAG DECISION: Database Good.")
        synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            text_qa_template=qa_prompt_tmpl
        )
        response = synthesizer.synthesize(user_query, nodes=reranked_nodes)
        return str(response)

    except Exception as e:
        print(f"❌ Error during AI processing: {e}")
        return "I encountered an internal error while processing your request."


def fallback_to_general_knowledge(query):
    print(f"🧠 Switching to General Knowledge...")
    try:
        response = llm.complete(query)
        return str(response)
    except Exception as e:
        print(f"❌ Error interacting with Ollama: {e}")
        return "I could not generate an answer. Is Ollama running?"


# --- MENU ACTIONS ---
def upload_document_console():
    print("\n📂 --- FILE UPLOAD ---")
    file_path = input("Paste file path: ").strip().replace('"', '')

    if not file_path:
        print("Cancelled.")
        return

    # Call the robust upload function
    model_db.upload_file(file_path, COLLECTION_NAME)

    # Reload index safely
    global index
    try:
        index = model_db.get_index(COLLECTION_NAME)
    except Exception as e:
        print(f"⚠️ Error reloading index: {e}")


def manage_documents_console():
    print("\n🗑️ --- MANAGE DOCUMENTS ---")
    try:
        files = model_db.list_uploaded_files(COLLECTION_NAME)
    except Exception:
        print("Could not retrieve file list.")
        return

    if not files:
        print("Database is empty or could not be reached.")
        return

    print("Current files in database:")
    for i, f in enumerate(files):
        print(f" [{i + 1}] {f}")

    print("\nType the number of the file to DELETE (or 'c' to cancel):")
    choice = input("Choice: ").strip()

    if choice.lower() == 'c':
        return

    # INPUT VALIDATION
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            file_to_delete = files[idx]
            confirm = input(f"Are you sure you want to delete '{file_to_delete}'? (y/n): ")
            if confirm.lower() == 'y':
                model_db.delete_file(file_to_delete, COLLECTION_NAME)

                # Reload index safely
                global index
                index = model_db.get_index(COLLECTION_NAME)
        else:
            print("❌ Invalid number selected.")
    except ValueError:
        print("❌ Invalid input. Please enter a number.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


# --- MAIN CONTROLLER ---
if __name__ == "__main__":
    while True:
        try:
            print("\n=== BACKEND MENU ===")
            print("1. Start Query Session")
            print("2. Upload Document")
            print("3. Manage Documents (Delete)")
            print("4. Exit")

            choice = input("Select: ").strip()

            if choice == "1":
                user_text = receive_user_query()
                if user_text:
                    final_answer = run_crag(user_text)
                    print(f"\n🤖 FINAL ANSWER:\n{final_answer}")
            elif choice == "2":
                upload_document_console()
            elif choice == "3":
                manage_documents_console()
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print("Invalid choice, please try again.")

        except KeyboardInterrupt:
            # Handles Ctrl+C gracefully
            print("\n\nProgram Interrupted. Exiting...")
            break
        except Exception as e:
            print(f"❌ Fatal Loop Error: {e}")