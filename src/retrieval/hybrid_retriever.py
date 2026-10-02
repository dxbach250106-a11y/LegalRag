from __future__ import annotations

import logging

import pandas as pd

from core.config import AppConfig
from models.embedder import QwenEmbeddingModel
from .rrf import ReciprocalRankFusion
from .schemas import RetrievalResult


class HybridRetriever:
    """BM25, vector and weighted RRF retrieval in one reusable class."""

    def __init__(self, config: AppConfig, embedder: QwenEmbeddingModel | None = None) -> None:
        self.config = config
        self.settings = config.section("retrieval")
        self.embedder = embedder or QwenEmbeddingModel(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.faiss_index = None
        self.faiss_metadata: pd.DataFrame | None = None
        self.bm25 = None
        self.bm25_lookup: pd.DataFrame | None = None

    def load(self) -> "HybridRetriever":
        import bm25s
        import faiss

        self.faiss_index = faiss.read_index(str(self.config.path("faiss_index")))
        self.faiss_metadata = pd.read_parquet(self.config.path("embedding_metadata"))
        self.bm25 = bm25s.BM25.load(str(self.config.path("bm25_dir")), load_corpus=False)
        self.bm25_lookup = pd.read_parquet(self.config.path("bm25_lookup"))
        if self.faiss_index.ntotal != len(self.faiss_metadata):
            raise ValueError("FAISS vector count and embedding metadata row count differ")
        return self

    def search_components(
        self,
        query: str,
        vector_k: int | None = None,
        bm25_k: int | None = None,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        """Compute vector and BM25 rankings once so tuning can reuse them."""
        if self.faiss_index is None:
            self.load()
        vector = self.search_vector(query, vector_k)
        bm25 = self.search_bm25(query, bm25_k)
        return vector, bm25

    def search_vector(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        if self.faiss_index is None:
            self.load()
        k = int(top_k or self.settings.get("faiss_top_k", 200))
        vector = self.embedder.encode([query]).astype("float32")
        scores, indices = self.faiss_index.search(vector, k)
        return [
            self._result(self.faiss_metadata.iloc[int(idx)], float(score), "vector")
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def search_bm25(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        if self.bm25 is None:
            self.load()
        import bm25s

        k = int(top_k or self.settings.get("bm25_top_k", 200))
        tokens = bm25s.tokenize([query], stopwords=None)
        indices, scores = self.bm25.retrieve(tokens, k=k)
        return [
            self._result(self.bm25_lookup.iloc[int(idx)], float(score), "bm25")
            for idx, score in zip(indices[0], scores[0])
        ]

    @staticmethod
    def fuse(
        vector_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        semantic_weight: float = 1.0,
        lexical_weight: float = 1.0,
        rrf_k: int = 60,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        fusion = ReciprocalRankFusion(int(rrf_k))
        fused = fusion.fuse(
            [
                (vector_results, float(semantic_weight)),
                (bm25_results, float(lexical_weight)),
            ]
        )
        return fused[:top_k] if top_k is not None else fused

    def search(
        self,
        query: str,
        top_k: int | None = None,
        semantic_weight: float | None = None,
        lexical_weight: float | None = None,
        rrf_k: int | None = None,
    ) -> list[RetrievalResult]:
        vector, bm25 = self.search_components(query)
        return self.fuse(
            vector,
            bm25,
            semantic_weight=float(self.settings.get("semantic_weight", 1.0) if semantic_weight is None else semantic_weight),
            lexical_weight=float(self.settings.get("lexical_weight", 1.0) if lexical_weight is None else lexical_weight),
            rrf_k=int(self.settings.get("rrf_k", 60) if rrf_k is None else rrf_k),
            top_k=int(top_k or self.settings.get("final_top_k", 200)),
        )

    def _result(self, row: pd.Series, score: float, source: str) -> RetrievalResult:
        id_col = self.config.get("embedding.id_column", "id")
        text_col = self.config.get("embedding.text_column", "content")
        metadata = row.to_dict()
        return RetrievalResult(
            str(row[id_col]).strip(),
            score,
            str(row.get(text_col, "")),
            source,
            metadata,
        )
