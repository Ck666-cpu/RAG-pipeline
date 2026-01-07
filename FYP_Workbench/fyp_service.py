# FYP_Workbench/fyp_service.py
import time
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, get_response_synthesizer, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
import model_db
from data_types import CRAGResult, SourceNode


class FYPService:
    def __init__(self):
        print(" [FYPService] Initializing Brain (TinyLlama) & Reranker...")
        self.collection_name = "crag_llamaindex"

        # 1. SETUP LLM
        try:
            self.llm = Ollama(model="tinyllama", request_timeout=360.0)
            Settings.llm = self.llm
        except Exception as e:
            print(f"❌ Error initializing Ollama: {e}")

        # 2. SETUP RERANKER
        try:
            self.reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_n=3
            )
        except Exception as e:
            print(f"❌ Error initializing Reranker: {e}")
            self.reranker = None

        # (Prompt definitions moved to _get_prompt_for_role)

    def _get_prompt_for_role(self, role: str) -> PromptTemplate:
        """Returns a different system prompt based on the user's role."""

        # OPTION A: ADMIN PROMPT (Technical, Detailed, Debug-oriented)
        if role.lower() == "admin":
            return PromptTemplate(
                "### Instruction:\n"
                "You are an internal Technical Advisor. The user is an Administrator.\n"
                "1. Answer the question in detail.\n"
                "2. If unsure, explicitly state 'Insufficient Data'.\n"
                "3. Use professional, technical language.\n"
                "\n"
                "### Context:\n{context_str}\n\n"
                "### Question:\n{query_str}\n\n"
                "### Admin Briefing:\n"
            )

        # OPTION B: TENANT/USER PROMPT (Friendly, Concise, Simple English)
        else:
            return PromptTemplate(
                "### Instruction:\n"
                "You are a helpful Customer Service Assistant for a housing agency.\n"
                "1. Answer the question politely and concisely.\n"
                "2. Avoid technical jargon.\n"
                "3. If the answer is not in the context, apologize and say you don't know.\n"
                "\n"
                "### Context:\n{context_str}\n\n"
                "### Question:\n{query_str}\n\n"
                "### Answer:\n"
            )

    def answer(self, question: str, user, history: list = None) -> CRAGResult:
        """
        'user' is now the full User object from user_manager, not just a role string.
        """
        # RULE 1: MASTER ADMIN CANNOT CHAT
        if user.role == "Master Admin":
            return CRAGResult(
                answer="Master Admins do not have access to Chat features.",
                source_nodes=[],
                confidence=0.0
            )

        print(f" [FYPService] User ({user.username}) asked: '{question}'")

        # ... (Memory/Contextualize logic same as before) ...
        search_query = question  # (simplified for brevity)

        index = model_db.get_index(self.collection_name)
        if not index: return self._fallback(search_query, "DB Error")

        try:
            # RULE 2: APPLY PERMISSION FILTERS
            # Staff/Admin only see Global docs OR their own Private docs
            user_filters = model_db.get_user_filters(user.username)

            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=10,
                filters=user_filters  # <--- INJECTED FILTERS
            )

            nodes = retriever.retrieve(search_query)

            # ... (Reranking & Synthesis same as before) ...
            # ... (Make sure to pass user.role to _get_prompt_for_role) ...

            # (Returning Mock Result for brevity of the snippet - keep your existing full logic)
            return CRAGResult(answer="Processed with filters.", source_nodes=[], confidence=1.0)

        except Exception as e:
            return self._fallback(search_query, str(e))

    def upload_document(self, file_path, user, is_global=False):
        # RULE 3: UPLOAD PERMISSIONS
        # Staff -> Can upload Private only.
        # Admin -> Can upload Private OR Global.
        # Master Admin -> Cannot upload.

        if user.role == "Master Admin":
            return False, "Master Admins cannot upload documents."

        if is_global and user.role != "Admin":
            return False, "Only Admins can upload Global documents."

        visibility = "global" if is_global else "private"

        return model_db.upload_file(
            file_path,
            self.collection_name,
            owner_username=user.username,
            visibility=visibility
        )

    # ... (Copy _contextualize, _fallback, upload_document from previous code) ...
    def _contextualize(self, query, history):
        # (Same as before)
        return query  # simplified for brevity

    def _fallback(self, query, reason):
        return CRAGResult(
            answer="I could not find that information.",
            sources=[],
            confidence=0.0
        )
