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

        # 1. SETUP LLM (TinyLlama for speed/memory safety)
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

        # 3. PROMPTS
        # A. Answer Prompt (Simplified for TinyLlama)
        self.qa_prompt = PromptTemplate(
            "### Instruction:\n"
            "You are a helpful assistant. Use the context below to answer the question concisely.\n"
            "If the answer is not in the context, say 'I cannot find that information'.\n"
            "Do not repeat these instructions.\n\n"
            "### Context:\n"
            "{context_str}\n\n"
            "### Question:\n"
            "{query_str}\n\n"
            "### Answer:\n"
        )

        # B. Memory/Rewrite Prompt (Simplified for TinyLlama)
        self.rewrite_prompt = PromptTemplate(
            "History of conversation:\n"
            "{history_str}\n\n"
            "User's new question: {query_str}\n\n"
            "Task: Rewrite the user's new question so it includes details from the history. "
            "Only output the new question. Do not explain.\n"
            "New Question: "
        )

    def answer(self, question: str, user_role: str, history: list = None) -> CRAGResult:
        """
        Now accepts 'history' - a list of previous strings (User, AI, User, AI...)
        """
        print(f" [FYPService] User asked: '{question}'")

        # 1. MEMORY CHECK: Contextualize if we have history
        search_query = question
        if history and len(history) > 0:
            print("   -> 🧠 Rewriting query using history...")
            search_query = self._contextualize(question, history)
            print(f"   -> 🔄 Rewritten as: '{search_query}'")

        # 2. LOAD INDEX
        index = model_db.get_index(self.collection_name)
        if not index:
            return self._fallback(search_query, "Database connection failed.")

            # Inside fyp_service.py -> answer() function

        try:
            # 3. RETRIEVE
            retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
            nodes = retriever.retrieve(search_query)

            if not nodes:
                return self._fallback(search_query, "No relevant docs found.")

            # --- DEBUG PRINT START: WHAT DID QDRANT FIND? ---
            print(f"\n[DEBUG] 1. Initial Retrieval found {len(nodes)} nodes.")
            for i, n in enumerate(nodes[:2]):  # Print first 2 only
                print(f"  - Node {i} Score: {n.score:.4f}")
                print(f"  - Content: {n.node.get_content()[:100]}...")  # First 100 chars
            # ------------------------------------------------

            # 4. RERANK
            if self.reranker:
                nodes = self.reranker.postprocess_nodes(nodes, query_str=search_query)

                # --- NEW CODE: FILTER OUT BAD MATCHES ---
                # Cross-Encoders usually output negative logits for "non-matches".
                # We filter anything below 0.0 (or -1.0 if you want to be lenient).
                nodes = [n for n in nodes if n.score > 0.0]

                if not nodes:
                    print("   -> ❌ Reranker filtered out all documents as irrelevant.")
                    return self._fallback(search_query, "Low relevance scores.")
                # ----------------------------------------

            # --- DEBUG PRINT START: WHAT SURVIVED RERANKING? ---
            print(f"\n[DEBUG] 2. After Reranking (Top {len(nodes)}):")
            for i, n in enumerate(nodes):
                print(f"  - Node {i} Score: {n.score:.4f}")
                print(f"  - Content: {n.node.get_content()[:100]}...")
            print("--------------------------------------------------\n")
            # ---------------------------------------------------

            # 5. GENERATE ANSWER
            synthesizer = get_response_synthesizer(
                response_mode="tree_summarize",
                text_qa_template=self.qa_prompt
            )
            response = synthesizer.synthesize(search_query, nodes=nodes)

            # ... rest of your code ...

            # 6. FORMAT RESULT
            sources = list(set([n.metadata.get('file_name', 'unknown') for n in nodes]))
            # Convert ugly logit score to a rough confidence % (sigmoid-ish)
            raw_score = nodes[0].score if nodes else 0.0
            confidence = 1 / (1 + 2.718 ** (-raw_score))  # Simple Math Sigmoid

            return CRAGResult(
                answer=str(response),
                sources=sources,
                confidence=float(confidence)
            )

        except Exception as e:
            print(f"❌ Error: {e}")
            return self._fallback(search_query, str(e))

    def _contextualize(self, query, history):
        """Uses LLM to rewrite 'It' or 'He' into specific names."""
        try:
            # Format history into a string
            history_str = ""
            recent_history = history[-2:]  # Only look at last 2 messages to keep it simple
            for msg in recent_history:
                history_str += f"- {msg}\n"

            # Ask LLM to rewrite
            response = self.llm.complete(
                self.rewrite_prompt.format(history_str=history_str, query_str=query)
            )

            # CLEANUP: TinyLlama often adds quotes or "Here is the question:"
            # We strip those out to get just the text.
            clean_text = str(response).strip().strip('"').strip("'")

            # If the result is super long (hallucination), ignore it and use original
            if len(clean_text) > len(query) + 100:
                return query

            return clean_text
        except Exception:
            return query

    def _fallback(self, query, reason):
        print(f"   -> Fallback triggered: {reason}")

        # NEW CODE: Return a static, honest message.
        return CRAGResult(
            answer="I could not find any specific information about that in the provided documents.",
            sources=["None (Low Relevance)"],
            confidence=0.0
        )

    # Added to help you upload files easily from test runner
    def upload_document(self, file_path):
        return model_db.upload_file(file_path, self.collection_name)