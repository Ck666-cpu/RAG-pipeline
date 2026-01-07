# FYP_Workbench/fyp_service.py
import time
from typing import Generator, Union
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

        # 3. PROMPTS
        # Prompt to rewrite "it" or "he" into specific names based on history
        self.rewrite_prompt = PromptTemplate(
            "History of conversation:\n{history_str}\n\n"
            "User's new question: {query_str}\n\n"
            "Task: Rewrite the user's new question to be a standalone query based on the history. "
            "If the user is just stating a fact, repeat the fact.\n"
            "New Question: "
        )

        # Prompt for General Chat (when no docs are found)
        # Simplified Prompt for TinyLlama
        self.general_chat_prompt = PromptTemplate(
            "You are a helpful assistant. Chat with the user politely.\n\n"
            "Conversation History:\n{history_str}\n\n"
            "User: {query_str}\n"
            "Assistant:"
        )

    def _get_prompt_for_role(self, role: str) -> PromptTemplate:
        """Returns a different system prompt based on the user's role."""
        if role.lower() == "admin":
            return PromptTemplate(
                "### Instruction:\n"
                "You are an internal Technical Advisor. The user is an Administrator.\n"
                "Answer based ONLY on the context below.\n"
                "### Context:\n{context_str}\n\n"
                "### Question:\n{query_str}\n\n"
                "### Answer:\n"
            )
        else:
            return PromptTemplate(
                "### Instruction:\n"
                "You are a helpful Customer Service Assistant.\n"
                "Answer based ONLY on the context below.\n"
                "### Context:\n{context_str}\n\n"
                "### Question:\n{query_str}\n\n"
                "### Answer:\n"
            )

    # CHANGED: Return type is now a Generator
    def answer(self, question: str, user, history: list = None) -> Generator[Union[CRAGResult, str], None, None]:

        # RULE 1: MASTER ADMIN CHECK
        if user.role == "Master Admin":
            yield CRAGResult(answer="Master Admins cannot chat.", confidence=0.0)
            return

        print(f" [FYPService] User ({user.username}) asked: '{question}'")

        # 1. REWRITE QUERY
        search_query = question
        if history and len(history) > 0:
            search_query = self._contextualize(question, history)

        # 2. RETRIEVE
        nodes = []
        try:
            index = model_db.get_index(self.collection_name)
            if index:
                user_filters = model_db.get_user_filters(user.username)
                retriever = VectorIndexRetriever(index=index, similarity_top_k=10, filters=user_filters)
                raw_nodes = retriever.retrieve(search_query)

                if raw_nodes and self.reranker:
                    nodes = self.reranker.postprocess_nodes(raw_nodes, query_str=search_query)
                    nodes = [n for n in nodes if n.score > 0.0]
        except Exception as e:
            print(f"   -> ⚠️ Retrieval Error: {e}")

        # --- STREAMING BRANCHES ---

        # OPTION A: RAG STREAMING
        if nodes:
            print(f"   -> ✅ Found {len(nodes)} docs. Streaming RAG...")

            # 1. Prepare Metadata (Sources)
            rich_sources = [
                SourceNode(
                    file_name=n.metadata.get('file_name', 'unknown'),
                    content_snippet=n.node.get_content()[:200] + "...",
                    score=float(n.score) if n.score else 0.0
                ) for n in nodes
            ]
            confidence = 1 / (1 + 2.718 ** (-nodes[0].score))

            # YIELD 1: Metadata Header
            yield CRAGResult(answer="", source_nodes=rich_sources, confidence=confidence)

            # 2. Start Streaming Text
            try:
                synthesizer = get_response_synthesizer(
                    response_mode="tree_summarize",
                    text_qa_template=self._get_prompt_for_role(user.role),
                    streaming=True  # <--- ENABLE STREAMING
                )
                response = synthesizer.synthesize(search_query, nodes=nodes)

                # YIELD 2+: Tokens
                for token in response.response_gen:
                    yield token

            except Exception as e:
                yield f"[Error: {str(e)}]"

        # OPTION B: MEMORY/CHAT STREAMING
        else:
            print("   -> 0 docs found. Streaming Chat Mode...")

            # YIELD 1: Metadata (Empty sources)
            yield CRAGResult(answer="", source_nodes=[], confidence=0.5)

            # 2. Stream from LLM directly
            try:
                history_str = "\n".join(history[-6:]) if history else "No history."
                prompt = self.general_chat_prompt.format(history_str=history_str, query_str=question)

                # Use stream_complete for raw text generation
                stream_gen = self.llm.stream_complete(prompt)

                for response_chunk in stream_gen:
                    yield response_chunk.delta

            except Exception as e:
                yield f"[Error: {str(e)}]"

    def _answer_from_memory(self, question, history):
        """Generates an answer using ONLY the chat history (No RAG)."""
        try:
            # Flatten history for the prompt (last 6 turns)
            history_str = "\n".join(history[-6:]) if history else "No history."

            # Ask LLM directly
            prompt = self.general_chat_prompt.format(
                history_str=history_str,
                query_str=question
            )
            response = self.llm.complete(prompt)

            return CRAGResult(
                answer=str(response),
                source_nodes=[],  # No sources for memory chat
                confidence=0.5  # Moderate confidence for general chat
            )
        except Exception as e:
            return self._fallback(question, str(e))

    def _contextualize(self, query, history):
        """Uses LLM to rewrite 'It' or 'He' into specific names."""
        try:
            history_str = "\n".join(history[-2:])  # Last 2 messages
            response = self.llm.complete(
                self.rewrite_prompt.format(history_str=history_str, query_str=query)
            )
            clean_text = str(response).strip().strip('"').strip("'")
            # Sanity check: if rewrite is too long, it's likely a hallucination
            if len(clean_text) > len(query) + 50:
                return query
            return clean_text
        except Exception:
            return query

    def _fallback(self, query, reason):
        print(f"   -> Fallback triggered: {reason}")
        return CRAGResult(
            answer="I could not find any specific information about that.",
            source_nodes=[],
            confidence=0.0
        )

    def upload_document(self, file_path, user, is_global=False):
        if user.role == "Master Admin":
            return False, "Master Admins cannot upload."
        if is_global and user.role != "Admin":
            return False, "Only Admins can upload Global docs."

        visibility = "global" if is_global else "private"
        return model_db.upload_file(
            file_path,
            self.collection_name,
            owner_username=user.username,
            visibility=visibility
        )