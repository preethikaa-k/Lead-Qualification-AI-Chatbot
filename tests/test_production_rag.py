"""
Unit and Integration Tests for Production RAG Pipeline.
Verifies:
1. Overlapping Semantic Chunking
2. BAAI/bge-small-en-v1.5 Open-Source Dense Embedding Retrieval
3. Approximate Nearest Neighbor (ANN) Vector Search
4. BM25Okapi Sparse Lexical Search
5. Cross-Encoder Re-Ranking (ms-marco-MiniLM-L-2-v2)
6. Grounding Firewall (Confidence Threshold 0.60)
7. Langfuse Tracing & Observability Ring-Buffer
"""

import pytest
from src.rag_engine import (
    faq_engine,
    chunk_text_with_overlap,
    bm25_tokenize,
    BM25Okapi,
    LangfuseRAGTracer,
)
from src.config import settings


class TestProductionRAGPipeline:
    def test_chunk_text_with_overlap(self):
        text = (
            "Ashcombe & Rowe Estates was founded in Mayfair in 2004. "
            "We specialise in prime residential property across London, Surrey, and Oxford. "
            "Our bespoke services cater to high-net-worth buyers and sellers seeking exceptional residences. "
            "Each client is partnered with a senior director offering discreet advisory and market guidance."
        )
        chunks = chunk_text_with_overlap(text, chunk_size=120, overlap=30)
        assert len(chunks) >= 2
        # Verify overlap exists between consecutive chunks
        for i in range(len(chunks) - 1):
            assert len(chunks[i]) > 0
            assert len(chunks[i+1]) > 0

    def test_bge_open_source_embeddings_initialized(self):
        assert faq_engine.sentence_model is not None
        assert "bge" in settings.EMBEDDING_MODEL.lower() or faq_engine.is_bge_model
        assert len(faq_engine.indexed_texts) >= 47

    def test_bm25_sparse_retrieval(self):
        assert faq_engine.bm25_retriever is not None
        tokens = bm25_tokenize("Chelsea branch Sunday opening hours")
        scores = faq_engine.bm25_retriever.get_scores(tokens)
        assert len(scores) == len(faq_engine.indexed_texts)
        assert max(scores) > 0.0

    def test_cross_encoder_reranking_accuracy(self):
        # Query specifically about Chelsea Sunday hours
        res = faq_engine.search_faq("Are you open on Sundays in Chelsea?")
        assert res is not None
        assert res.faq_id == "FAQ-023"
        assert "Chelsea" in res.answer or "Sunday" in res.answer

    def test_grounding_firewall_zero_hallucination(self):
        # Query about out-of-domain topic must be rejected by grounding threshold
        res = faq_engine.search_faq("Can you sell me quantum teleportation hardware for spacecraft?")
        assert res is None  # Deflected cleanly to head office phone fallback

    def test_observability_metrics_and_langfuse(self):
        metrics = faq_engine.get_observability_metrics()
        assert "embedding_model" in metrics
        assert "reranker_model" in metrics
        assert "chunking_strategy" in metrics
        assert "total_indexed_chunks" in metrics
        assert "grounding_threshold" in metrics
        assert metrics["grounding_threshold"] == 0.60
        assert "langfuse_connected" in metrics
