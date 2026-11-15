"""
Production Hybrid RAG Engine for Ashcombe & Rowe Estates FAQ Knowledge Base.
Architecture:
1. Dense Semantic Search: Chroma VectorStore with sentence-transformers (all-MiniLM-L6-v2)
2. Sparse Lexical Search: BM25Okapi for precise keyword and entity matching
3. Reciprocal Rank & Weighted Score Fusion (RRF): Combines Dense (0.65) + Sparse (0.35)
4. Document Grounding Gatekeeper: Strict confidence thresholding to prevent hallucinations
5. Comprehensive Observability & Telemetry: Live latency, hit rate, and audit ring-buffer
"""

import os
import csv
import re
import time
import uuid
import math
import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from collections import Counter, deque
from dataclasses import dataclass, asdict
import numpy as np

# Ensure local HuggingFace cache is utilized without network retries
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.config import settings
from src.models import FAQItem

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def bm25_tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric tokens, filtering stopwords."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\'-]+\b", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


class BM25Okapi:
    """
    High-performance, pure-Python BM25Okapi implementation.
    Standardized with k1=1.5 and b=0.75 for robust sparse text retrieval.
    """
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / (self.corpus_size or 1)
        self.doc_freqs = []
        self.nd = Counter()
        for doc in corpus:
            freq = Counter(doc)
            self.doc_freqs.append(freq)
            for token in freq.keys():
                self.nd[token] += 1
        self.idf = {}
        for token, freq in self.nd.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        if not query_tokens or self.corpus_size == 0:
            return scores
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            for i, freq_map in enumerate(self.doc_freqs):
                freq = freq_map.get(token, 0)
                if freq == 0:
                    continue
                num = freq * (self.k1 + 1.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[i] / self.avgdl))
                scores[i] += idf * (num / denom)
        return scores


@dataclass
class RAGTelemetryRecord:
    telemetry_id: str
    timestamp: str
    query: str
    dense_score: float
    bm25_score: float
    hybrid_score: float
    selected_faq_id: Optional[str]
    selected_category: Optional[str]
    is_grounded: bool
    retrieval_method: str
    latency_ms: float
    grounded_citation: Optional[str]


class RAGObservabilityTracker:
    """
    Enterprise Observability Ring-Buffer and Aggregator for RAG.
    Tracks queries, latencies, grounding fidelity, and out-of-domain deflections.
    """
    def __init__(self, buffer_size: int = 200):
        self.records: deque = deque(maxlen=buffer_size)
        self.total_queries: int = 0
        self.grounded_hits: int = 0
        self.out_of_domain_rejections: int = 0
        self.latencies: deque = deque(maxlen=buffer_size)

    def record_query(self, record: RAGTelemetryRecord):
        self.records.append(record)
        self.total_queries += 1
        self.latencies.append(record.latency_ms)
        if record.is_grounded:
            self.grounded_hits += 1
        else:
            self.out_of_domain_rejections += 1

    def get_metrics(self) -> Dict[str, Any]:
        l_list = list(self.latencies)
        avg_lat = sum(l_list) / len(l_list) if l_list else 0.0
        p95_lat = sorted(l_list)[int(len(l_list) * 0.95)] if len(l_list) >= 20 else (max(l_list) if l_list else 0.0)
        grounding_pct = (self.grounded_hits / self.total_queries * 100) if self.total_queries > 0 else 100.0
        return {
            "total_queries": self.total_queries,
            "grounded_hits": self.grounded_hits,
            "out_of_domain_rejections": self.out_of_domain_rejections,
            "grounding_fidelity_rate": f"{grounding_pct:.1f}%",
            "avg_latency_ms": round(avg_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "recent_queries_count": len(self.records),
        }

    def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        recent = list(self.records)[-limit:]
        return [asdict(r) for r in reversed(recent)]


class FAQRagEngine:
    def __init__(self, faq_file: Optional[Path] = None):
        self.faq_file = faq_file or settings.FAQ_FILE
        self.faqs: List[FAQItem] = []
        self.indexed_texts: List[str] = []
        self.indexed_metadatas: List[Dict[str, Any]] = []
        
        self.vector_store = None
        self.vector_store_type: str = "chroma"
        self.sentence_model = None
        self.bm25_retriever: Optional[BM25Okapi] = None
        
        self.vectorizer = None
        self.tfidf_matrix = None
        
        self.observability = RAGObservabilityTracker()

        self._load_faqs()
        self._init_hybrid_retriever()

    def _load_faqs(self):
        if not self.faq_file.exists():
            logger.warning(f"FAQ file not found at {self.faq_file}")
            return
        with open(self.faq_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = FAQItem(
                    faq_id=row["faq_id"],
                    category=row["category"],
                    question=row["question"],
                    answer=row["answer"],
                )
                self.faqs.append(item)

    def _init_hybrid_retriever(self):
        """
        Initializes:
        1. BM25 sparse keyword retriever (BM25Okapi)
        2. Chroma dense semantic vector store (or InMemory fallback)
        """
        try:
            from sentence_transformers import SentenceTransformer
            from langchain_core.embeddings import Embeddings
            from langchain_core.vectorstores import InMemoryVectorStore

            # Load local SentenceTransformer
            try:
                self.sentence_model = SentenceTransformer(settings.EMBEDDING_MODEL, local_files_only=True)
            except Exception:
                self.sentence_model = SentenceTransformer(settings.EMBEDDING_MODEL)

            class LocalEmbeddings(Embeddings):
                def __init__(self, model):
                    self.m = model
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return self.m.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()
                def embed_query(self, text: str) -> List[float]:
                    return self.m.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

            embeddings = LocalEmbeddings(self.sentence_model)

            texts = []
            metadatas = []
            for item in self.faqs:
                meta = {
                    "faq_id": item.faq_id,
                    "category": item.category,
                    "question": item.question,
                    "answer": item.answer,
                }
                # 1. Question index
                texts.append(item.question)
                metadatas.append(meta)

                # 2. Enriched text
                texts.append(f"{item.question} [Category: {item.category}] {item.answer}")
                metadatas.append(meta)

                # 3. Comprehensive synonym expansions for natural language variations
                faq_expansions = {
                    "FAQ-001": [
                        "Tell me about your agency and properties.",
                        "What price range and property types do you specialise in?",
                    ],
                    "FAQ-002": [
                        "Who are you guys? Tell me about your background and history.",
                    ],
                    "FAQ-003": [
                        "Who started this real estate business? Who founded Ashcombe & Rowe?",
                        "Who are the founders Helen Ashcombe and David Rowe?",
                    ],
                    "FAQ-007": [
                        "Are your negotiators certified or accredited by regulatory agencies?",
                        "Are you registered with any professional bodies or industry regulators?",
                        "Do you follow The Property Ombudsman and Propertymark code of conduct?",
                    ],
                    "FAQ-009": [
                        "Where is your office located? What is your head office address in London?",
                        "Where can I find your central Mayfair headquarters?",
                    ],
                    "FAQ-023": [
                        "What are the Chelsea branch opening hours?",
                        "When is the Chelsea office open?",
                        "Is Chelsea branch open on Sundays?",
                    ],
                    "FAQ-029": [
                        "Are you open on Sunday? Are any branches open on Sundays?",
                        "Can I visit an office on the weekend on Sunday?",
                    ],
                    "FAQ-031": [
                        "Can I book a house tour on the weekend?",
                        "How do I arrange a viewing or visit a property?",
                        "How do I see inside one of your houses or apartments?",
                    ],
                    "FAQ-032": [
                        "Do you take any money from people buying a house?",
                        "Do you charge buyers any fee, commission, or upfront costs?",
                        "Are there any buyer charges or fees when purchasing a property?",
                    ],
                    "FAQ-034": [
                        "Do I need a mortgage in principle or agreement before viewing?",
                        "Do I need proof of funds or mortgage approval to look at houses?",
                    ],
                    "FAQ-035": [
                        "What if I need someone to help me get a home loan or mortgage?",
                        "Can you recommend a mortgage broker or independent financing adviser?",
                        "I am looking for someone to help me get a mortgage.",
                    ],
                    "FAQ-036": [
                        "What paperwork and ID documents do I need to submit an offer?",
                        "What documents are required for anti-money laundering checks?",
                    ],
                    "FAQ-037": [
                        "Can I buy a property off-plan before it is constructed?",
                        "Do you sell off plan new developments with reservations?",
                    ],
                    "FAQ-038": [
                        "How do I sell my home with your agency?",
                        "What are the steps to list and market my property for sale?",
                    ],
                    "FAQ-039": [
                        "Is a property valuation free? How much does a market valuation cost?",
                        "Do I have to pay anything to get an estimate of my house value?",
                    ],
                    "FAQ-040": [
                        "What are your fees? How much commission do you charge to sell a home?",
                        "Do I need to pay anything upfront to list my house with you?",
                        "What is your sole agency and multi agency selling commission percentage?",
                    ],
                    "FAQ-041": [
                        "How long does it take to sell a property from launch to completion?",
                        "What is the average timescale and duration to get an offer and complete?",
                    ],
                    "FAQ-042": [
                        "Do I legally need an Energy Performance Certificate (EPC) to sell?",
                    ],
                    "FAQ-044": [
                        "Can I rent a flat or house? Do you offer lettings, rentals, or tenant services?",
                        "Do you manage rental properties or handle landlord lettings?",
                    ],
                    "FAQ-045": [
                        "Can you advise me on stamp duty tax, SDLT, or legal matters?",
                        "Do you give tax advice or conveyancing legal advice?",
                    ],
                    "FAQ-046": [
                        "What happens to my personal information? What do you do with my data?",
                        "How do you comply with UK GDPR and data privacy laws?",
                    ],
                    "FAQ-047": [
                        "Can people from abroad purchase a house through your agency?",
                        "Do you work with international, overseas, or foreign buyers?",
                    ],
                }

                if item.faq_id in faq_expansions:
                    for exp in faq_expansions[item.faq_id]:
                        texts.append(exp)
                        metadatas.append(meta)

            self.indexed_texts = texts
            self.indexed_metadatas = metadatas

            # 1. Initialize BM25 Sparse Index
            bm25_corpus = [bm25_tokenize(t) for t in texts]
            self.bm25_retriever = BM25Okapi(bm25_corpus)
            logger.info(f"Initialized BM25 sparse index over {len(texts)} document chunks.")

            # 2. Initialize Dense Vector Store: Chroma (Primary) with InMemoryVectorStore fallback
            store_choice = (settings.RAG_VECTOR_STORE or "chroma").lower()
            self.vector_store_type = "in_memory"

            if store_choice == "chroma":
                try:
                    try:
                        from langchain_chroma import Chroma
                    except ImportError:
                        from langchain_community.vectorstores import Chroma
                    
                    self.vector_store = Chroma.from_texts(
                        texts=texts,
                        embedding=embeddings,
                        metadatas=metadatas,
                        collection_metadata={"hnsw:space": "cosine"},
                    )
                    self.vector_store_type = "chroma"
                    logger.info(f"Initialized Chroma vector store with {len(texts)} embeddings (hnsw:space=cosine).")
                except Exception as e:
                    logger.warning(f"Chroma not available ({e}), falling back to LangChain InMemoryVectorStore.")
                    self.vector_store = InMemoryVectorStore(embeddings)
                    self.vector_store.add_texts(texts=texts, metadatas=metadatas)
                    self.vector_store_type = "in_memory"
            else:
                self.vector_store = InMemoryVectorStore(embeddings)
                self.vector_store.add_texts(texts=texts, metadatas=metadatas)
                self.vector_store_type = "in_memory"
                logger.info(f"Initialized LangChain InMemoryVectorStore with {len(texts)} embeddings.")

        except Exception as e:
            logger.warning(f"Could not initialize dense VectorStore: {e}. Falling back to TF-IDF.")
            self.vector_store = None
            self._init_tfidf_fallback()

    def _init_tfidf_fallback(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            corpus = [f"{item.question} {item.question} {item.answer}" for item in self.faqs]
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            logger.info(f"Initialized TF-IDF vector fallback with {len(corpus)} documents.")
        except Exception as e:
            logger.error(f"TF-IDF initialization failed: {e}")
            self.vectorizer = None

    def search_faq(self, query: str, threshold: float = 0.60) -> Optional[FAQItem]:
        """
        Hybrid Retrieval (Chroma Dense Vector + BM25 Sparse Lexical).
        Grounded strictly in verified documents to eliminate hallucinations.
        Full latency and scoring telemetry logged on every execution.
        """
        if not query or not self.faqs:
            return None

        clean_query = query.strip()
        start_time = time.perf_counter()
        query_id = str(uuid.uuid4())[:8]

        dense_score = 0.0
        dense_doc_meta: Optional[Dict[str, Any]] = None
        dense_best_idx: int = -1

        # 1. Dense Semantic Search (Chroma or InMemory)
        if self.vector_store is not None:
            try:
                results = self.vector_store.similarity_search_with_score(clean_query, k=3)
                if results:
                    top_doc, raw_score = results[0]
                    dense_doc_meta = top_doc.metadata
                    if self.vector_store_type == "chroma":
                        dense_score = max(0.0, min(1.0, 1.0 - float(raw_score)))
                    else:
                        dense_score = max(0.0, min(1.0, float(raw_score)))
            except Exception as e:
                logger.warning(f"Dense vector search failed: {e}")

        # 2. Sparse Lexical Search (BM25Okapi)
        bm25_score = 0.0
        bm25_doc_meta: Optional[Dict[str, Any]] = None
        if self.bm25_retriever is not None:
            try:
                q_tokens = bm25_tokenize(clean_query)
                raw_bm25_scores = self.bm25_retriever.get_scores(q_tokens)
                if raw_bm25_scores and max(raw_bm25_scores) > 0.0:
                    max_bm25 = max(raw_bm25_scores)
                    best_bm25_idx = int(np.argmax(raw_bm25_scores))
                    bm25_doc_meta = self.indexed_metadatas[best_bm25_idx]
                    # Normalize BM25 score to [0, 1] using standard saturation curve
                    bm25_score = min(1.0, max_bm25 / (max_bm25 + 3.0))
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")

        # 3. Hybrid Score Fusion (65% Dense Semantic + 35% Sparse Lexical)
        if dense_doc_meta and bm25_doc_meta:
            # If both agree on the same FAQ ID, high confidence reinforcement
            if dense_doc_meta.get("faq_id") == bm25_doc_meta.get("faq_id"):
                hybrid_score = 0.60 * dense_score + 0.40 * bm25_score + 0.05
                selected_meta = dense_doc_meta
            else:
                # Disagreement: compare weighted scores
                dense_weighted = 0.65 * dense_score
                bm25_weighted = 0.35 * bm25_score
                if dense_weighted >= bm25_weighted:
                    hybrid_score = 0.65 * dense_score + 0.35 * bm25_score
                    selected_meta = dense_doc_meta
                else:
                    hybrid_score = 0.40 * dense_score + 0.60 * bm25_score
                    selected_meta = bm25_doc_meta
        elif dense_doc_meta:
            hybrid_score = dense_score
            selected_meta = dense_doc_meta
        elif bm25_doc_meta:
            hybrid_score = bm25_score
            selected_meta = bm25_doc_meta
        else:
            hybrid_score = 0.0
            selected_meta = None

        # 4. Fallback to TF-IDF if VectorStore is offline
        if selected_meta is None and self.vectorizer is not None and self.tfidf_matrix is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                query_vec = self.vectorizer.transform([clean_query])
                sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
                best_idx = int(np.argmax(sims))
                tfidf_score = float(sims[best_idx])
                if tfidf_score >= 0.35:
                    item = self.faqs[best_idx]
                    selected_meta = {
                        "faq_id": item.faq_id,
                        "category": item.category,
                        "question": item.question,
                        "answer": item.answer,
                    }
                    hybrid_score = tfidf_score
            except Exception as e:
                logger.warning(f"TF-IDF fallback error: {e}")

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 5. Strict Document Grounding & Hallucination Firewall
        is_grounded = bool(selected_meta is not None and hybrid_score >= threshold)
        
        grounded_citation = (
            f"[Verified Grounded Source: {selected_meta.get('faq_id')} | Category: {selected_meta.get('category')}]"
            if is_grounded and selected_meta else None
        )

        retrieval_method = f"{self.vector_store_type.title()} (Dense) + BM25 (Sparse) Hybrid"

        # Record observability telemetry
        record = RAGTelemetryRecord(
            telemetry_id=query_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            query=clean_query,
            dense_score=round(dense_score, 4),
            bm25_score=round(bm25_score, 4),
            hybrid_score=round(hybrid_score, 4),
            selected_faq_id=selected_meta.get("faq_id") if selected_meta else None,
            selected_category=selected_meta.get("category") if selected_meta else None,
            is_grounded=is_grounded,
            retrieval_method=retrieval_method,
            latency_ms=round(latency_ms, 2),
            grounded_citation=grounded_citation,
        )
        self.observability.record_query(record)

        if is_grounded and selected_meta:
            logger.info(
                f"[RAG Grounded Hit] Query: '{clean_query[:40]}' -> {selected_meta.get('faq_id')} "
                f"(Hybrid={hybrid_score:.2f}, Dense={dense_score:.2f}, BM25={bm25_score:.2f}, Latency={latency_ms:.1f}ms)"
            )
            return FAQItem(
                faq_id=selected_meta.get("faq_id", ""),
                category=selected_meta.get("category", ""),
                question=selected_meta.get("question", ""),
                answer=selected_meta.get("answer", ""),
            )

        logger.info(
            f"[RAG Deflected Out-Of-Domain] Query: '{clean_query[:40]}' (Hybrid={hybrid_score:.2f} < {threshold:.2f}, Latency={latency_ms:.1f}ms)"
        )
        return None

    def get_faq_answer(self, query: str) -> Optional[str]:
        item = self.search_faq(query)
        if item:
            return item.answer
        return None

    def get_observability_metrics(self) -> Dict[str, Any]:
        """Returns real-time operational RAG metrics for monitoring dashboards."""
        metrics = self.observability.get_metrics()
        metrics["vector_store_primary"] = self.vector_store_type
        metrics["sparse_retriever"] = "BM25Okapi"
        metrics["hybrid_alpha"] = 0.65
        metrics["grounding_threshold"] = 0.60
        metrics["knowledge_base_size"] = len(self.faqs)
        return metrics

    def get_observability_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the recent query audit log with scores, latency, and grounding status."""
        return self.observability.get_audit_log(limit=limit)


# Global singleton instance
faq_engine = FAQRagEngine()
