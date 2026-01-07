# FYP_Workbench/fyp_service.py
import time
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, get_response_synthesizer, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank

import model_db
from data_types import CRAGResult


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

    # ... (Rewrite Prompt remains the same or can also be moved) ...

    def answer(self, question: str, user_role: str, history: list = None) -> CRAGResult:
        print(f" [FYPService] User ({user_role}) asked: '{question}'")

        # 1. MEMORY CHECK
        search_query = question
        if history and len(history) > 0:
            search_query = self._contextualize(question, history)

        # 2. LOAD INDEX
        index = model_db.get_index(self.collection_name)
        if not index:
            return self._fallback(search_query, "Database connection failed.")

        try:
            # 3. RETRIEVE
            retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
            nodes = retriever.retrieve(search_query)

            if not nodes:
                return self._fallback(search_query, "No relevant docs found.")

            # 4. RERANK
            if self.reranker:
                nodes = self.reranker.postprocess_nodes(nodes, query_str=search_query)
                nodes = [n for n in nodes if n.score > 0.0]  # Filter

                if not nodes:
                    return self._fallback(search_query, "Low relevance scores.")

            # 5. GENERATE ANSWER (DYNAMIC PROMPT HERE)
            # Fetch the specific prompt for this role
            role_prompt = self._get_prompt_for_role(user_role)

            synthesizer = get_response_synthesizer(
                response_mode="tree_summarize",
                text_qa_template=role_prompt  # <--- INJECTED HERE
            )
            response = synthesizer.synthesize(search_query, nodes=nodes)

            # 6. FORMAT RESULT
            sources = list(set([n.metadata.get('file_name', 'unknown') for n in nodes]))
            raw_score = nodes[0].score if nodes else 0.0
            confidence = 1 / (1 + 2.718 ** (-raw_score))

            return CRAGResult(
                answer=str(response),
                sources=sources,
                confidence=float(confidence)
            )

        except Exception as e:
            print(f"❌ Error: {e}")
            return self._fallback(search_query, str(e))

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

    def upload_document(self, file_path):
        return model_db.upload_file(file_path, self.collection_name)