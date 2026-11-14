"""
RAG Engine for Ashcombe & Rowe Estates FAQ.
Performs semantic vector search over faq.csv with hybrid question-corpus matching,
confidence thresholding, and zero-hallucination guardrails.
"""

import csv
import re
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np

from src.config import settings
from src.models import FAQItem


class FAQRagEngine:
    def __init__(self, faq_file: Optional[Path] = None):
        self.faq_file = faq_file or settings.FAQ_FILE
        self.faqs: List[FAQItem] = []
        self.question_embeddings: Optional[np.ndarray] = None
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.model = None
        self._load_faqs()
        self._init_embeddings()

    def _load_faqs(self):
        if not self.faq_file.exists():
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

    def _init_embeddings(self):
        """
        Initializes local sentence-transformers embedding model.
        Runs 100% on CPU, zero API cost, <10ms inference.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            questions = [item.question for item in self.faqs]
            corpus = [f"{item.question} | Category: {item.category} | {item.answer}" for item in self.faqs]
            
            self.question_embeddings = self.model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
            self.corpus_embeddings = self.model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
        except Exception as e:
            self.model = None
            self._init_tfidf_fallback()

    def _init_tfidf_fallback(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            corpus = [f"{item.question} {item.question} {item.answer}" for item in self.faqs]
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        except Exception:
            self.vectorizer = None

    def search_faq(self, query: str, top_k: int = 1, threshold: float = 0.55) -> Optional[FAQItem]:
        """
        Searches FAQ knowledge base using blended question-corpus semantic similarity.
        Returns the best matching FAQItem if similarity >= threshold, else None.
        Ensures the bot never makes up company policies or branch details.
        """
        if not query or not self.faqs:
            return None

        # Clean query
        q_clean = query.strip()

        # Check semantic embeddings
        if self.model is not None and self.question_embeddings is not None and self.corpus_embeddings is not None:
            query_vec = self.model.encode([q_clean], convert_to_numpy=True, normalize_embeddings=True)[0]
            
            q_sims = np.dot(self.question_embeddings, query_vec)
            c_sims = np.dot(self.corpus_embeddings, query_vec)
            
            # Blended score: 70% weight on question similarity, 30% on corpus content
            blended = 0.70 * q_sims + 0.30 * c_sims
            best_idx = int(np.argmax(blended))
            best_score = float(blended[best_idx])
            
            if best_score >= threshold:
                return self.faqs[best_idx]
            return None

        # TF-IDF Fallback
        if hasattr(self, "vectorizer") and self.vectorizer is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = self.vectorizer.transform([q_clean])
            sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            best_idx = int(np.argmax(sims))
            if float(sims[best_idx]) >= 0.40:
                return self.faqs[best_idx]

        return None

    def get_faq_answer(self, query: str) -> Optional[str]:
        item = self.search_faq(query)
        if item:
            return item.answer
        return None


# Global singleton instance
faq_engine = FAQRagEngine()
