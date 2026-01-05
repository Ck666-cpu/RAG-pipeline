# FYP_Workbench/fyp_service.py
import time
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, get_response_synthesizer, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank

# --- LOCAL WORKBENCH IMPORTS ---
import model_db
from data_types import CRAGResult


class FYPService:
    def __init__(self):
        print(" [FYPService] Initializing Brain (Phi-3) & Reranker...")
        self.collection_name = "crag_llamaindex"

        # 1. SETUP LLM (Ollama)
        try:
            self.llm = Ollama(model="phi3", request_timeout=360.0)
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

        # 3. PROMPT
        self.qa_prompt = PromptTemplate(
            "Context:\n{context_str}\n\n"
            "Question: {query_str}\n"
            "Answer (be concise): "
        )

    def upload_document(self, file_path: str):
        """Helper to upload files from the service."""
        print(f" [FYPService] Uploading: {file_path}")
        success, msg = model_db.upload_file(file_path, self.collection_name)
        return success, msg

    def answer(self, question: str, user_role: str) -> CRAGResult:
        print(f" [FYPService] Thinking about: '{question}'...")

        # A. LOAD INDEX (Refresh every time to catch new uploads)
        index = model_db.get_index(self.collection_name)
        if not index:
            return self._fallback(question, "Database connection failed or empty.")

        try:
            # B. RETRIEVE
            retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
            nodes = retriever.retrieve(question)

            if not nodes:
                return self._fallback(question, "No relevant docs found.")

            # C. RERANK
            if self.reranker:
                nodes = self.reranker.postprocess_nodes(nodes, query_str=question)

            # D. GENERATE
            synthesizer = get_response_synthesizer(
                response_mode="tree_summarize",
                text_qa_template=self.qa_prompt
            )
            response = synthesizer.synthesize(question, nodes=nodes)

            # E. EXTRACT SOURCES
            sources = list(set([n.metadata.get('file_name', 'unknown') for n in nodes]))
            confidence = nodes[0].score if nodes else 0.0

            return CRAGResult(
                answer=str(response),
                sources=sources,
                confidence=float(confidence)
            )

        except Exception as e:
            print(f"❌ Error: {e}")
            return self._fallback(question, str(e))

    def _fallback(self, query, reason):
        """If RAG fails, ask pure LLM."""
        print(f"   -> Fallback triggered: {reason}")
        resp = self.llm.complete(query)
        return CRAGResult(answer=str(resp), sources=["General Knowledge"], confidence=0.1)