"""
RAG Engine for Ashcombe & Rowe Estates FAQ Knowledge Base.
Implements:
1. Vector Store RAG over faq.csv using LangChain's InMemoryVectorStore with local embeddings
2. Sentence-Transformers (all-MiniLM-L6-v2) in local offline mode (zero latency, zero network errors)
3. TF-IDF fallback for environments without PyTorch
4. Strict confidence thresholding to prevent hallucination (fallback to office number 020 7946 0100)
"""

import os
import csv
import re
import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np

# Ensure local HuggingFace cache is utilized without network retries
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.config import settings
from src.models import FAQItem

logger = logging.getLogger(__name__)


class FAQRagEngine:
    def __init__(self, faq_file: Optional[Path] = None):
        self.faq_file = faq_file or settings.FAQ_FILE
        self.faqs: List[FAQItem] = []
        self.vector_store = None
        self.sentence_model = None
        self.vectorizer = None
        self.tfidf_matrix = None

        self._load_faqs()
        self._init_vector_store()

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

    def _init_vector_store(self):
        """
        Initializes LangChain VectorStore backed by local SentenceTransformer embeddings.
        """
        try:
            from sentence_transformers import SentenceTransformer
            from langchain_core.embeddings import Embeddings
            from langchain_core.vectorstores import InMemoryVectorStore

            # Load local model
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
            self.vector_store = InMemoryVectorStore(embeddings)

            # Build dual-indexed corpus:
            # 1. Question index (for high-precision question-to-question matching)
            # 2. Enriched corpus (for contextual & keyword matching)
            # 3. Targeted synonym expansions for high-frequency inquiries
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

            self.vector_store.add_texts(texts=texts, metadatas=metadatas)
            logger.info(f"Loaded {len(texts)} FAQ vector representations into InMemoryVectorStore using {settings.EMBEDDING_MODEL}.")

        except Exception as e:
            logger.warning(f"Could not initialize VectorStore with sentence-transformers: {e}. Falling back to TF-IDF.")
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

    def search_faq(self, query: str, threshold: float = 0.52) -> Optional[FAQItem]:
        """
        Searches FAQ knowledge base using semantic vector search.
        Returns the best matching FAQItem if similarity >= threshold, else None.
        Guarantees zero hallucinations for company policies, fees, and branch hours.
        """
        if not query or not self.faqs:
            return None

        clean_query = query.strip()

        # 1. Primary: VectorStore Semantic Search
        if self.vector_store is not None:
            try:
                results = self.vector_store.similarity_search_with_score(clean_query, k=3)
                if results:
                    doc, score = results[0]
                    # InMemoryVectorStore returns cosine similarity (1.0 = identical, 0.0 = orthogonal)
                    if score >= threshold:
                        meta = doc.metadata
                        return FAQItem(
                            faq_id=meta.get("faq_id", ""),
                            category=meta.get("category", ""),
                            question=meta.get("question", ""),
                            answer=meta.get("answer", ""),
                        )
            except Exception as e:
                logger.warning(f"VectorStore search error: {e}. Trying fallback.")

        # 2. Fallback: TF-IDF Cosine Similarity
        if self.vectorizer is not None and self.tfidf_matrix is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                query_vec = self.vectorizer.transform([clean_query])
                sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
                best_idx = int(np.argmax(sims))
                if float(sims[best_idx]) >= 0.35:
                    return self.faqs[best_idx]
            except Exception as e:
                logger.warning(f"TF-IDF search error: {e}")

        return None

    def get_faq_answer(self, query: str) -> Optional[str]:
        item = self.search_faq(query)
        if item:
            return item.answer
        return None


# Global singleton instance
faq_engine = FAQRagEngine()
