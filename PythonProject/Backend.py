import sys
import model_db
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, get_response_synthesizer, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.llms import ChatMessage, MessageRole

# --- CONFIGURATION ---
COLLECTION_NAME = "crag_llamaindex"
chat_history = []  # <--- GLOBAL MEMORY LIST

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
    sys.exit(1)

# 2. SETUP PROMPT (Strict Guide)
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
try:
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=1
    )
except Exception as e:
    print(f"❌ Error downloading Reranker: {e}")
    sys.exit(1)

# 4. SETUP DATABASE
index = None
try:
    index = model_db.get_index(COLLECTION_NAME)
except Exception:
    pass


# --- MEMORY FUNCTION: CONTEXTUALIZER ---
def contextualize_query(new_query):
    """
    If there is history, ask LLM to rewrite the query to include context.
    If no history, just return the query as is.
    """
    if not chat_history:
        return new_query

    # Create a temporary prompt just for rewriting
    rewrite_prompt = (
        "You are a helpful assistant. "
        "The user has asked a follow-up question based on the chat history below.\n"
        "Rephrase the follow-up question to be a standalone question that can be understood without the history.\n"
        "Do NOT answer the question. Just rewrite it.\n\n"
        "Chat History:\n"
    )

    # Add last 2 exchanges to save RAM
    for msg in chat_history[-4:]:
        rewrite_prompt += f"{msg.role}: {msg.content}\n"

    rewrite_prompt += f"\nFollow-up Question: {new_query}\n"
    rewrite_prompt += "Standalone Question:"

    try:
        response = llm.complete(rewrite_prompt)
        rewritten_query = str(response).strip()
        print(f"🔄 Rewritten Query: '{rewritten_query}'")
        return rewritten_query
    except Exception as e:
        print(f"⚠️ Rewrite failed: {e}")
        return new_query


# --- CRAG LOGIC ---
def run_crag(user_query):
    if not index:
        return "System Error: Please upload a document first."

    # 1. REWRITE QUERY (Memory Step)
    search_query = contextualize_query(user_query)

    print(f"⚙️ Processing: '{search_query}'")

    try:
        # 2. RETRIEVE
        retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
        retrieved_nodes = retriever.retrieve(search_query)

        if not retrieved_nodes:
            return fallback_to_general_knowledge(search_query)

        # 3. RERANK
        reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_str=search_query)

        if not reranked_nodes:
            return fallback_to_general_knowledge(search_query)

        # 4. EVALUATE
        best_score = reranked_nodes[0].score
        print(f"   -> Relevance Score: {best_score:.4f}")

        if best_score < 0.2:
            print("🔴 CRAG DECISION: Database Irrelevant.")
            return fallback_to_general_knowledge(search_query)

        # 5. GENERATE
        print("🟢 CRAG DECISION: Database Good.")
        synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            text_qa_template=qa_prompt_tmpl
        )
        response = synthesizer.synthesize(search_query, nodes=reranked_nodes)

        # SAVE TO MEMORY
        save_history(user_query, str(response))

        return str(response)

    except Exception as e:
        print(f"❌ Error: {e}")
        return "Internal Processing Error."


def fallback_to_general_knowledge(query):
    print(f"🧠 Switching to General Knowledge...")
    response = llm.complete(query)
    save_history(query, str(response))
    return str(response)


def save_history(user_text, ai_text):
    """Saves the interaction to the global list"""
    chat_history.append(ChatMessage(role=MessageRole.USER, content=user_text))
    chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=ai_text))


def reset_chat():
    global chat_history
    chat_history = []
    print("🧹 Chat history cleared.")


# --- MENU ACTIONS ---
def receive_user_query():
    print("\n" + "-" * 30)
    return input("❓ Enter your query: ").strip()


def upload_document_console():
    print("\n📂 --- FILE UPLOAD ---")
    file_path = input("Paste file path: ").strip().replace('"', '')
    if file_path:
        model_db.upload_file(file_path, COLLECTION_NAME)
        global index
        index = model_db.get_index(COLLECTION_NAME)
    else:
        print("Cancelled.")


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
        print("❌ Invalid input.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


# --- MAIN CONTROLLER ---
if __name__ == "__main__":
    while True:
        try:
            print("\n=== BACKEND MENU ===")
            print("1. Chat (Query)")
            print("2. Reset Memory")
            print("3. Upload Document")
            print("4. Manage Documents (Delete)")
            print("5. Exit")

            choice = input("Select: ").strip()

            if choice == "1":
                user_text = receive_user_query()
                if user_text:
                    final_answer = run_crag(user_text)
                    print(f"\n🤖 FINAL ANSWER:\n{final_answer}")
            elif choice == "2":
                reset_chat()
            elif choice == "3":
                upload_document_console()
            elif choice == "4":
                manage_documents_console()
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Invalid choice.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")