import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import BaseModel, ConfigDict, Field
from rank_bm25 import BM25Okapi

from .log_utils import get_logger

logger = get_logger(__name__)


class _LibItem(BaseModel):
    module_name: str
    json_obj: Dict[str, Any]
    text: str
    vector: Optional[np.ndarray] = None

    # pydantic v2: allow numpy arrays
    model_config = ConfigDict(arbitrary_types_allowed=True)


def _flatten_lib_json(j: Dict[str, Any]) -> str:
    """Turn a library module json into a searchable text blob."""
    parts: List[str] = []
    parts.append(f"module_name: {j.get('module_name','')}")
    desc = j.get("description", {})
    if isinstance(desc, dict):
        if "function" in desc:
            parts.append(f"function: {desc['function']}")
        if "protocols" in desc:
            parts.append(f"protocols: {desc['protocols']}")
    elif isinstance(desc, str):
        parts.append(f"description: {desc}")
    kws = j.get("keywords", [])
    if isinstance(kws, list):
        parts.append("keywords: " + ", ".join(kws))
    # index parameters/ports names for structure/context
    for p in j.get("parameters", []) or []:
        parts.append(f"param {p.get('name','')}: {p.get('type','')}")
    for p in j.get("ports", []) or []:
        parts.append(
            f"port {p.get('direction','')} {p.get('name','')}: {p.get('width','')}"
        )
    return "\n".join(parts)


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


class LibConsultantOutput(BaseModel):
    reuse: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    consult: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class LibConsultant:
    """
    Deterministic library search over module JSONs using a HYBRID retriever:
      - Dense vectors (FAISS, cosine via inner product on L2-normalized embeddings)
      - Lexical BM25 (rank_bm25)

    Consumes a planner JSON and returns the same structure with concrete library hits.
    """

    def __init__(self, embed_model: BaseEmbedding):
        self.embed_model = embed_model
        self.items: List[_LibItem] = []
        # Dense (FAISS)
        self.faiss_index: Optional[faiss.Index] = None
        self.dim: Optional[int] = None
        self._mat: Optional[np.ndarray] = None
        # Lexical (BM25)
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus_tokens: List[List[str]] = []
        self._texts: List[str] = []

    def ingest_from_dir(self, lib_dir: str, pattern: str = "**/*.json") -> int:
        """
        Recursively ingest JSON files under lib_dir.
        Returns number of items ingested.
        """
        paths = glob.glob(os.path.join(lib_dir, pattern), recursive=True)
        loaded = 0
        for p in paths:
            try:
                with open(p, "r") as f:
                    j = json.load(f)
                name = j.get("module_name") or os.path.splitext(os.path.basename(p))[0]
                text = _flatten_lib_json(j)
                self.items.append(_LibItem(module_name=name, json_obj=j, text=text))
                self._texts.append(text)
                loaded += 1
            except Exception as e:
                logger.warning(f"LibConsultant Skipping {p}: {e}")
        logger.info(f"LibConsultant Ingested {loaded} modules from {lib_dir}")
        return loaded

    def ingest_from_list(self, modules: Sequence[Dict[str, Any]]) -> int:
        loaded = 0
        for j in modules:
            try:
                name = j.get("module_name", "unknown")
                text = _flatten_lib_json(j)
                self.items.append(_LibItem(module_name=name, json_obj=j, text=text))
                self._texts.append(text)
                loaded += 1
            except Exception as e:
                logger.warning(f"LibConsultant Skipping in-memory item: {e}")
        logger.info(f"LibConsultant Ingested {loaded} in-memory modules")
        return loaded

    def build_index(self) -> None:
        """
        Build FAISS (dense) and BM25 (lexical) indexes.
        """
        if not self.items:
            logger.warning("LibConsultant No items to index.")
            return

        # Dense embeddings (FAISS)
        vectors: List[np.ndarray] = []
        for it in self.items:
            emb = self.embed_model.get_text_embedding(it.text)
            v = np.array(emb, dtype=np.float32)
            v /= np.linalg.norm(v) + 1e-12
            it.vector = v
            vectors.append(v)

        if vectors:
            mat = np.vstack(vectors)
            self._mat = mat
            self.dim = mat.shape[1]
            self.faiss_index = faiss.IndexFlatIP(self.dim)
            self.faiss_index.add(mat)
            logger.info(
                f"LibConsultant FAISS index built. dim={self.dim}, n={len(self.items)}"
            )
        else:
            self.faiss_index = None
            self._mat = None
            self.dim = None
            logger.warning("LibConsultant No vectors to add to FAISS.")

        # Lexical tokens (BM25)
        def _tok(s: str) -> List[str]:
            # Minimal tokenizer; swap for a smarter one if desired.
            return s.lower().split()

        self._bm25_corpus_tokens = [_tok(t) for t in self._texts]
        if self._bm25_corpus_tokens:
            self._bm25 = BM25Okapi(self._bm25_corpus_tokens)
            logger.info(f"LibConsultant BM25 index built. n={len(self.items)}")
        else:
            self._bm25 = None
            logger.warning("LibConsultant No texts for BM25.")

    def _search_dense(
        self, query_text: str, fanout: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (idxs, scores) for dense search.
        """
        n = len(self.items)
        if n == 0 or self.faiss_index is None:
            return np.array([], dtype=int), np.array([], dtype=float)

        qv = np.array(self.embed_model.get_text_embedding(query_text), dtype=np.float32)
        qv = _normalize_rows(_ensure_2d(qv))
        k = min(fanout, n)
        distances, indices = self.faiss_index.search(qv, k)
        idxs = indices[0]
        scores = distances[0]  # cosine similarity in [-1,1]
        mask = idxs >= 0
        return idxs[mask], scores[mask].astype(float)

    def _search_lexical(
        self, query_text: str, fanout: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (idxs, scores) for BM25 search.
        """
        n = len(self.items)
        if n == 0 or self._bm25 is None:
            return np.array([], dtype=int), np.array([], dtype=float)

        toks = query_text.lower().split()
        scores = self._bm25.get_scores(toks)  # arbitrary non-negative BM25 scores
        cand = np.argsort(scores)[::-1][: min(fanout, n)]
        return cand.astype(int), scores[cand].astype(float)

    @staticmethod
    def _rrf_fuse(
        n: int, ranks_a: Dict[int, int], ranks_b: Dict[int, int], k: float = 60.0
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion (RRF): score = 1/(k + rank_a) + 1/(k + rank_b).
        Ranks are 1-based. Items missing from a list get no contribution from that list.
        """
        fused: Dict[int, float] = {}
        for i, r in ranks_a.items():
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + r)
        for i, r in ranks_b.items():
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + r)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        mn, mx = float(x.min()), float(x.max())
        if mx <= mn:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    def _search_hybrid(
        self,
        query_text: str,
        top_k: int = 5,
        fanout: int = 5,
        method: str = "rrf",
        alpha: float = 0.65,
    ) -> List[Tuple[int, float]]:
        """
        Hybrid search combining dense (FAISS) and lexical (BM25).
        Returns list of (index, fused_score) sorted by score desc.
        """
        n = len(self.items)
        if n == 0:
            return []

        dense_idxs, dense_scores = self._search_dense(
            query_text, fanout=max(fanout, top_k)
        )
        bm25_idxs, bm25_scores = self._search_lexical(
            query_text, fanout=max(fanout, top_k)
        )

        if method == "rrf":
            # Build rank maps (1-based ranks)
            ranks_dense: Dict[int, int] = {
                int(i): int(r + 1) for r, i in enumerate(dense_idxs)
            }
            ranks_bm25: Dict[int, int] = {
                int(i): int(r + 1) for r, i in enumerate(bm25_idxs)
            }
            fused = self._rrf_fuse(n, ranks_dense, ranks_bm25, k=3.0)
            return [(idx, float(score)) for idx, score in fused[:top_k] if score > 0.0]

        # Weighted-sum fusion (score normalization)
        s_dense = np.zeros(n, dtype=np.float32)
        s_bm25 = np.zeros(n, dtype=np.float32)
        s_dense[dense_idxs] = dense_scores
        s_bm25[bm25_idxs] = bm25_scores

        s_dense = self._minmax(s_dense)
        s_bm25 = self._minmax(s_bm25)
        fused = alpha * s_dense + (1.0 - alpha) * s_bm25

        order = np.argsort(fused)[::-1][:top_k]
        return [(int(i), float(fused[i])) for i in order if fused[i] > 0.0]

    def _best_hit_json(
        self,
        query_text: str,
        score_threshold: float,
        top_k: int,
        fanout: int = 5,
        method: str = "rrf",
        alpha: float = 0.65,
    ) -> Optional[Dict[str, Any]]:

        hits = self._search_hybrid(
            query_text=query_text,
            top_k=top_k,
            fanout=fanout,
            method=method,
            alpha=alpha,
        )
        if not hits:
            return None
        idx, score = hits[0]
        # For RRF, scores are small positives; keep threshold semantics simple.
        if score < score_threshold:
            return None
        return self.items[idx].json_obj

    def consult(
        self,
        design_plan: Dict[str, Any],
        score_threshold: float = 0.49,
        top_k: int = 1,
        *,
        fanout: int = 5,
        method: str = "rrf",  # "rrf" or "weighted"
        alpha: float = 0.65,  # used if method == "weighted"
    ) -> str:
        """
        Returns json_str containing library matches for both 'reuse' and 'consult'.

        Args:
          score_threshold: threshold on fused score (set to 0.0 to disable).
          top_k: number of final results to consider (we always pick the best for library_json).
          fanout: candidates taken from each retriever before fusion (usually >= top_k).
          method: "rrf" for Reciprocal Rank Fusion, or "weighted" for normalized weighted sum.
          alpha: weight for dense scores in "weighted" fusion (ignored for "rrf").
        """

        out = LibConsultantOutput(reuse={}, consult={})

        for top_key in ("reuse", "consult"):
            bucket = design_plan.get(top_key, {}) or {}
            if not isinstance(bucket, dict):
                continue

            for key, val in bucket.items():
                if not isinstance(val, dict):
                    continue

                desc = val.get("description", "") or ""
                kws = val.get("keywords", []) or []
                protos = val.get("protocols", []) or []
                reasoning = val.get("reasoning", "") or ""

                query_text = (
                    f"{top_key} | {desc}\n"
                    f"keywords: {', '.join(kws) if isinstance(kws, list) else str(kws)}\n"
                    f"protocols: {', '.join(protos) if isinstance(protos, list) else str(protos)}\n"
                    f"reasoning: {reasoning}"
                )

                lib_json = self._best_hit_json(
                    query_text,
                    score_threshold=score_threshold,
                    top_k=max(1, top_k),
                    fanout=max(fanout, top_k),
                    method=method,
                    alpha=alpha,
                )
                entry = {"library_json": lib_json}  # may be None if no confident match
                getattr(out, top_key)[key] = entry

        json_str = json.dumps(out.dict(), indent=2)
        return json_str
