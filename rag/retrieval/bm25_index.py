from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple, Optional

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_PHRASE_RE = re.compile(r'"([^"]+)"')


def raw_tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75

    # Query-term-frequency control.
    # Larger k3 means repeated query terms matter more.
    k3: float = 1.2

    # Phrase boost for exact normalized phrase matches.
    phrase_boost: float = 1.5

    # Stopword removal
    remove_stopwords: bool = True

    # Stemming
    use_stemming: bool = True

    # Optional synonym expansion weight
    synonym_weight: float = 0.3


class TextPreprocessor:
    def __init__(
        self,
        *,
        remove_stopwords: bool = True,
        use_stemming: bool = True,
        synonyms: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.stopwords = set(ENGLISH_STOP_WORDS) if remove_stopwords else set()
        self.stemmer = PorterStemmer() if use_stemming else None
        self.synonyms = synonyms or {}

    def normalize_token(self, token: str) -> str:
        token = token.lower()
        if self.use_stemming and self.stemmer is not None:
            token = self.stemmer.stem(token)
        return token

    def tokenize(self, text: str) -> List[str]:
        tokens = raw_tokenize(text)
        out: List[str] = []

        for tok in tokens:
            if self.remove_stopwords and tok in self.stopwords:
                continue
            out.append(self.normalize_token(tok))

        return out

    def expand_query_terms(self, terms: List[str]) -> Dict[str, float]:
        """
        Returns term -> query weight.
        Base terms get weight 1.0 each time they appear.
        Synonyms get lower weight.
        """
        weighted_terms: Dict[str, float] = {}

        for term in terms:
            weighted_terms[term] = weighted_terms.get(term, 0.0) + 1.0

            for syn in self.synonyms.get(term, []):
                norm_syn = self.normalize_token(syn)
                if norm_syn != term:
                    weighted_terms[norm_syn] = weighted_terms.get(norm_syn, 0.0) + 0.3

        return weighted_terms


@dataclass
class BM25Index:
    """
    Persisted BM25 index over chunks.

    Stores:
      - doc_len[chunk_id] : int
      - postings[term][chunk_id] : tf (int)
      - df[term] : int
      - N, avgdl
      - doc_terms[chunk_id] : normalized token list (for phrase matching)
    """
    config: BM25Config
    postings: Dict[str, Dict[str, int]]
    doc_len: Dict[str, int]
    df: Dict[str, int]
    doc_terms: Dict[str, List[str]]
    N: int
    avgdl: float
    corpus_fingerprint: str
    synonyms: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.preprocessor = TextPreprocessor(
            remove_stopwords=self.config.remove_stopwords,
            use_stemming=self.config.use_stemming,
            synonyms=self.synonyms,
        )

    @staticmethod
    def build(
        chunks: Iterable[Tuple[str, str]],  # (chunk_id, text)
        *,
        config: BM25Config = BM25Config(),
        corpus_fingerprint: str,
        synonyms: Optional[Dict[str, List[str]]] = None,
    ) -> "BM25Index":
        postings: Dict[str, Dict[str, int]] = {}
        doc_len: Dict[str, int] = {}
        df: Dict[str, int] = {}
        doc_terms: Dict[str, List[str]] = {}

        preprocessor = TextPreprocessor(
            remove_stopwords=config.remove_stopwords,
            use_stemming=config.use_stemming,
            synonyms=synonyms,
        )

        N = 0
        total_len = 0

        for chunk_id, text in chunks:
            N += 1
            tokens = preprocessor.tokenize(text)
            doc_terms[chunk_id] = tokens

            L = len(tokens)
            doc_len[chunk_id] = L
            total_len += L

            tf = Counter(tokens)

            for term, freq in tf.items():
                postings.setdefault(term, {})[chunk_id] = freq

            for term in tf.keys():
                df[term] = df.get(term, 0) + 1

        avgdl = (total_len / N) if N > 0 else 0.0

        return BM25Index(
            config=config,
            postings=postings,
            doc_len=doc_len,
            df=df,
            doc_terms=doc_terms,
            N=N,
            avgdl=avgdl,
            corpus_fingerprint=corpus_fingerprint,
            synonyms=synonyms or {},
        )

    def _idf(self, term: str) -> float:
        n_qi = self.df.get(term, 0)
        if n_qi == 0:
            return 0.0
        return math.log(1.0 + (self.N - n_qi + 0.5) / (n_qi + 0.5))

    def _query_tf_weight(self, qtf: float) -> float:
        """
        BM25-style query term frequency factor.
        """
        k3 = self.config.k3
        return ((k3 + 1.0) * qtf) / (k3 + qtf) if qtf > 0 else 0.0

    def _contains_phrase(self, doc_tokens: List[str], phrase_tokens: List[str]) -> bool:
        if not phrase_tokens or len(phrase_tokens) > len(doc_tokens):
            return False

        m = len(phrase_tokens)
        for i in range(len(doc_tokens) - m + 1):
            if doc_tokens[i:i + m] == phrase_tokens:
                return True
        return False

    def _extract_phrase_tokens(self, query: str) -> List[List[str]]:
        phrases = _PHRASE_RE.findall(query)
        out: List[List[str]] = []

        for phrase in phrases:
            ptoks = self.preprocessor.tokenize(phrase)
            if ptoks:
                out.append(ptoks)

        return out

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not query.strip():
            return []

        # Main processed query terms
        query_terms = self.preprocessor.tokenize(query)
        if not query_terms:
            return []

        # Query term frequency + synonym expansion
        weighted_query_terms = self.preprocessor.expand_query_terms(query_terms)
        phrase_tokens_list = self._extract_phrase_tokens(query)

        k1 = self.config.k1
        b = self.config.b

        scores: Dict[str, float] = {}

        for term, raw_qtf in weighted_query_terms.items():
            idf = self._idf(term)
            if idf == 0.0:
                continue

            plist = self.postings.get(term)
            if not plist:
                continue

            q_weight = self._query_tf_weight(raw_qtf)

            for chunk_id, tf in plist.items():
                dl = self.doc_len.get(chunk_id, 0)

                denom = tf + k1 * (
                    1.0 - b + b * (dl / self.avgdl if self.avgdl > 0 else 0.0)
                )

                doc_term_score = idf * ((tf * (k1 + 1.0)) / (denom if denom != 0 else 1.0))
                scores[chunk_id] = scores.get(chunk_id, 0.0) + (q_weight * doc_term_score)

        # Phrase boost
        if phrase_tokens_list:
            for chunk_id in list(scores.keys()):
                dterms = self.doc_terms.get(chunk_id, [])
                phrase_hits = sum(
                    1 for phrase_tokens in phrase_tokens_list
                    if self._contains_phrase(dterms, phrase_tokens)
                )
                if phrase_hits > 0:
                    scores[chunk_id] *= (1.0 + self.config.phrase_boost * phrase_hits)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        payload = {
            "config": {
                "k1": self.config.k1,
                "b": self.config.b,
                "k3": self.config.k3,
                "phrase_boost": self.config.phrase_boost,
                "remove_stopwords": self.config.remove_stopwords,
                "use_stemming": self.config.use_stemming,
                "synonym_weight": self.config.synonym_weight,
            },
            "N": self.N,
            "avgdl": self.avgdl,
            "corpus_fingerprint": self.corpus_fingerprint,
            "doc_len": self.doc_len,
            "df": self.df,
            "postings": self.postings,
            "doc_terms": self.doc_terms,
            "synonyms": self.synonyms,
        }

        with open(os.path.join(dir_path, "bm25.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @staticmethod
    def load(dir_path: str) -> "BM25Index":
        with open(os.path.join(dir_path, "bm25.json"), "r", encoding="utf-8") as f:
            payload = json.load(f)

        cfg = payload["config"]

        return BM25Index(
            config=BM25Config(
                k1=float(cfg["k1"]),
                b=float(cfg["b"]),
                k3=float(cfg.get("k3", 1.2)),
                phrase_boost=float(cfg.get("phrase_boost", 1.5)),
                remove_stopwords=bool(cfg.get("remove_stopwords", True)),
                use_stemming=bool(cfg.get("use_stemming", True)),
                synonym_weight=float(cfg.get("synonym_weight", 0.3)),
            ),
            postings={
                k: {kk: int(vv) for kk, vv in v.items()}
                for k, v in payload["postings"].items()
            },
            doc_len={k: int(v) for k, v in payload["doc_len"].items()},
            df={k: int(v) for k, v in payload["df"].items()},
            doc_terms={k: list(v) for k, v in payload["doc_terms"].items()},
            N=int(payload["N"]),
            avgdl=float(payload["avgdl"]),
            corpus_fingerprint=str(payload["corpus_fingerprint"]),
            synonyms=payload.get("synonyms", {}),
        )