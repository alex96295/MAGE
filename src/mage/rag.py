from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import faiss
from llama_index.core import (
    Document,
    PropertyGraphIndex,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.indices.property_graph import VectorContextRetriever
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever

# Neo4j property graph store (optional)
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.legacy.indices.knowledge_graph.retrievers import KGRetrieverMode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore

from .log_utils import get_logger

logger = get_logger(__name__)


@dataclass
class VectorIndexSettings:
    """Settings for a vector index."""

    persist_dir: str
    faiss_path: str = None
    embedding_dim: int = 1024
    embed_model: BaseEmbedding = None


@dataclass
class GraphIndexSettings:
    """Settings for a knowledge graph index (Neo4j)."""

    persist_dir: str
    neo4j_url: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str = "neo4j"  # default DB name


@dataclass
class ChatTemplates:
    """Optional custom prompts for ContextChatEngine."""

    context_template: Optional[object] = None  # PromptTemplate
    refine_template: Optional[object] = None  # PromptTemplate


class VectorIndexManager:
    """Builds or loads a single VectorStoreIndex for a corpus."""

    def __init__(self, settings: VectorIndexSettings):
        self.settings = settings
        os.makedirs(self.settings.persist_dir, exist_ok=True)
        if self.settings.faiss_path:
            os.makedirs(os.path.dirname(self.settings.faiss_path), exist_ok=True)

    def build_or_load(self, documents: Sequence[Document]) -> VectorStoreIndex:
        """Create a FAISS-backed index or load from disk if present."""
        # If a persisted index exists -> load it
        if self.settings.faiss_path and os.path.exists(self.settings.faiss_path):
            logger.info("Loading existing persisted vector store and index")
            vector_store = FaissVectorStore.from_persist_path(self.settings.faiss_path)
            storage_context = StorageContext.from_defaults(
                persist_dir=self.settings.persist_dir,
                vector_store=vector_store,
            )
            index = load_index_from_storage(storage_context)
            logger.info("Loaded existing FAISS vector index.")
            return index

        # Otherwise, build from documents
        logger.info("No existing vector store/index found. Creating new one...")
        vector_store = None
        logger.info(faiss.__version__)
        logger.info(faiss.get_compile_options())
        faiss_index = faiss.IndexFlatIP(self.settings.embedding_dim)
        logger.info("Debug")
        logger.info(faiss_index)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        logger.info(vector_store)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            list(documents),
            storage_context=storage_context,
            embed_model=self.settings.embed_model,
            show_progress=True,
            use_async=True,
        )

        # Persist both the storage and FAISS index (if path provided)
        logger.info("Persisting newly created vector index")
        vector_store.persist(persist_path=self.settings.faiss_path)
        index.storage_context.persist(persist_dir=self.settings.persist_dir)
        logger.info(
            f"Persisted FAISS vector index to {self.settings.faiss_path} and storage to {self.settings.persist_dir}."
        )
        return index


class GraphIndexManager:
    """Builds or loads a PropertyGraphIndex (Neo4j-backed)."""

    def __init__(self, settings: GraphIndexSettings):
        self.settings = settings
        os.makedirs(self.settings.persist_dir, exist_ok=True)

    def _neo4j_store(self) -> Neo4jPropertyGraphStore:
        return Neo4jPropertyGraphStore(
            url=self.settings.neo4j_url,
            username=self.settings.neo4j_user,
            password=self.settings.neo4j_password,
            database=self.settings.neo4j_database,
        )

    def build_or_load(self, documents: Sequence[Document]) -> PropertyGraphIndex:
        graph_store = self._neo4j_store()

        # Try load from storage first
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=self.settings.persist_dir,
                graph_store=graph_store,
            )
            index = load_index_from_storage(storage_context)
            return index
        except Exception:
            # Fall back to building a fresh graph index
            pass

        index = PropertyGraphIndex.from_documents(
            list(documents),
            llm=Settings.llm,
            embed_model=Settings.embed_model,
            storage_context=StorageContext.from_defaults(graph_store=graph_store),
            property_graph_store=graph_store,
            show_progress=True,
            use_async=True,
        )
        index.storage_context.persist(persist_dir=self.settings.persist_dir)
        # Note: graph_store is external (Neo4j), no local persistence to write.
        return index


class RetrieverFactory:
    """Factory for creating various retrievers from your indexes."""

    # ---- Vector retriever ----
    @staticmethod
    def vector(index: VectorStoreIndex, top_k: int = 2) -> BaseRetriever:
        return index.as_retriever(similarity_top_k=top_k)

    # ---- BM25 retriever ----
    @staticmethod
    def bm25_from_index(index: VectorStoreIndex, top_k: int = 2) -> BM25Retriever:
        """
        BM25 doesn't need its own 'index' object; it works from a docstore.
        Using the docstore from your VectorStoreIndex is convenient.
        """
        return BM25Retriever.from_defaults(
            docstore=index.docstore, similarity_top_k=top_k
        )

    @staticmethod
    def bm25_from_graph(
        graph_index: PropertyGraphIndex, top_k: int = 2
    ) -> BM25Retriever:
        return BM25Retriever.from_defaults(
            docstore=graph_index.docstore, similarity_top_k=top_k
        )

    # ---- Graph retriever ----
    @staticmethod
    def graph(
        index: PropertyGraphIndex,
        mode: KGRetrieverMode = KGRetrieverMode.HYBRID,
        top_k: int = 2,
        num_chunks_per_query: int = 5,
        max_keywords_per_query: int = 10,
    ) -> BaseRetriever:
        """
        For knowledge-graph querying. HYBRID typically mixes embedding + keyword paths.
        """
        return index.as_retriever(
            retriever_mode=mode,
            similarity_top_k=top_k,
            num_chunks_per_query=num_chunks_per_query,
            max_keywords_per_query=max_keywords_per_query,
        )

    # ---- Graph + Vector context retriever ----
    @staticmethod
    def graph_with_vector_context(
        index: PropertyGraphIndex,
        top_k: int = 2,
        path_depth: int = 10,
        include_text: bool = True,
    ) -> BaseRetriever:
        """
        Augment graph retriever with a vector context retriever backed by the *graph's* store.
        This is useful when the graph store supports vectors (like Neo4j genAI plugin).
        """
        vec_ctx = VectorContextRetriever(
            index.property_graph_store,
            embed_model=Settings.embed_model,
            include_text=include_text,
            similarity_top_k=top_k,
            path_depth=path_depth,
        )
        return index.as_retriever(sub_retrievers=[vec_ctx])

    # ---- Fusion retriever ----
    @staticmethod
    def fusion(
        retrievers: Iterable[BaseRetriever],
        top_k: int = 2,
        num_queries: int = 4,
        mode: str = "reciprocal_rerank",
        use_async: bool = True,
        verbose: bool = False,
    ) -> QueryFusionRetriever:
        """
        Fuse multiple retrievers (e.g., vector + BM25 + graph).
        mode: 'reciprocal_rerank' | 'relative_score' | 'dist_based_score' | 'simple'
        """
        return QueryFusionRetriever(
            list(retrievers),
            similarity_top_k=top_k,
            num_queries=num_queries,
            mode=mode,
            use_async=use_async,
            verbose=verbose,
        )


class ChatEngineFactory:
    """
    Builds ContextChatEngine(s) from a retriever + LLM.
    Memory is optional; if omitted, engines are effectively stateless
    (similar to plain llm.chat(messages) if you already pass prior turns).
    """

    @staticmethod
    def text_engine(
        retriever: BaseRetriever,
        llm: Optional[LLM] = None,
        memory: Optional[ChatMemoryBuffer] = None,
        templates: Optional[ChatTemplates] = None,
    ) -> ContextChatEngine:
        return ContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm or Settings.llm,
            memory=memory,
            context_template=(templates.context_template if templates else None),
            context_refine_template=(templates.refine_template if templates else None),
        )

    @staticmethod
    def code_engine(
        retriever: BaseRetriever,
        code_llm: Optional[LLM] = None,
        memory: Optional[ChatMemoryBuffer] = None,
        templates: Optional[ChatTemplates] = None,
    ) -> ContextChatEngine:
        return ContextChatEngine.from_defaults(
            retriever=retriever,
            llm=code_llm or Settings.code_llm or Settings.llm,
            memory=memory,
            context_template=(templates.context_template if templates else None),
            context_refine_template=(templates.refine_template if templates else None),
        )


def create_vector_retriever_from_docs(
    documents: Sequence[Document],
    v_settings: VectorIndexSettings,
    top_k: int = 2,
) -> tuple[VectorStoreIndex, BaseRetriever]:
    """
    Build/load a vector index and return (index, retriever).
    """
    v_manager = VectorIndexManager(v_settings)
    v_index = v_manager.build_or_load(documents)
    v_retriever = RetrieverFactory.vector(v_index, top_k=top_k)
    return v_index, v_retriever


def create_graph_retriever_from_docs(
    documents: Sequence[Document],
    g_settings: GraphIndexSettings,
    top_k: int = 2,
    mode: KGRetrieverMode = KGRetrieverMode.HYBRID,
) -> tuple[PropertyGraphIndex, BaseRetriever]:
    """
    Build/load a graph index and return (index, graph retriever).
    """
    g_manager = GraphIndexManager(g_settings)
    g_index = g_manager.build_or_load(documents)
    g_retriever = RetrieverFactory.graph(g_index, mode=mode, top_k=top_k)
    return g_index, g_retriever


def create_fusion_retriever(
    base_retrievers: Iterable[BaseRetriever],
    top_k: int = 2,
    num_queries: int = 4,
    mode: str = "reciprocal_rerank",
    use_async: bool = True,
    verbose: bool = False,
) -> QueryFusionRetriever:
    """Create a fusion retriever from a list of retrievers."""
    return RetrieverFactory.fusion(
        retrievers=base_retrievers,
        top_k=top_k,
        num_queries=num_queries,
        mode=mode,
        use_async=use_async,
        verbose=verbose,
    )
