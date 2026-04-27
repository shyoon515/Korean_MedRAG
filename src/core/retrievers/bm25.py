"""
BM25 기반 Sparse Retriever
"""
import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import logging
import re
from tqdm import tqdm

try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    print("Warning: kiwipiepy not installed. Install with: pip install kiwipiepy")
    KIWI_AVAILABLE = False

from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """한국어 형태소 분석 기반 BM25 Retriever"""
    
    def __init__(
        self,
        use_kiwi: bool = True,
        strict_kiwi: bool = True,
        logger: logging.Logger = None,
    ):
        """
        Args:
            use_kiwi: kiwipiepy 사용 여부
            strict_kiwi: True이면 형태소 분석기 미사용 상태에서 실패
        """
        self.bm25 = None
        self.chunks = None
        self.tokenizer = None
        self.logger = logger or logging.getLogger(__name__)
        self.strict_kiwi = strict_kiwi
        self.use_kiwi = use_kiwi and KIWI_AVAILABLE
        
        if self.use_kiwi:
            self._init_tokenizer()
        else:
            if use_kiwi and not KIWI_AVAILABLE:
                message = "[BM25] kiwipiepy 미설치. 형태소 분석이 필수입니다. pip install kiwipiepy"
                if self.strict_kiwi:
                    raise RuntimeError(message)
                self.logger.warning("%s (fallback tokenizer 사용)", message)
    
    def _init_tokenizer(self):
        """Kiwi 토크나이저 초기화"""
        try:
            self.tokenizer = Kiwi()
            self.logger.info("[BM25] Kiwi 토크나이저 초기화 완료")
        except Exception as e:
            self.logger.exception("[BM25] Kiwi 초기화 실패")
            self.tokenizer = None
            self.use_kiwi = False
            if self.strict_kiwi:
                raise RuntimeError("[BM25] Kiwi 초기화 실패. 형태소 분석 필수 모드라 중단합니다.") from e
    
    def _tokenize(self, text: str) -> List[str]:
        """텍스트 토크나이제이션"""
        normalized = self._normalize_text(text)

        if self.use_kiwi and self.tokenizer is not None:
            try:
                tokens = self.tokenizer.tokenize(normalized)

                # 의료 QA에서 노이즈를 줄이기 위해 정보량이 높은 품사 위주로 사용
                allowed_tags = {"NNG", "NNP", "SL", "SN", "SH", "XR"}
                filtered = [tok.form.lower() for tok in tokens if tok.tag in allowed_tags and len(tok.form) >= 2]

                # 필터링 결과가 비면 전체 토큰으로 복구
                if filtered:
                    return filtered
                return [tok.form.lower() for tok in tokens if len(tok.form) >= 2]
            except Exception as e:
                if self.strict_kiwi:
                    raise RuntimeError("[BM25] 형태소 분석 오류. 형태소 분석 필수 모드라 중단합니다.") from e
                self.logger.warning("[BM25] 형태소 분석 오류. 띄어쓰기 토크나이저 사용")
                self.use_kiwi = False
        
        # Fallback: 한글/영문/숫자 토큰만 추출
        fallback_tokens = re.findall(r"[가-힣A-Za-z0-9]+", normalized.lower())
        return [t for t in fallback_tokens if len(t) >= 2]

    def _normalize_text(self, text: str) -> str:
        """BM25 검색 노이즈를 줄이기 위한 기본 전처리."""
        t = text.replace("\n", " ")

        # 객관식 번호 패턴 제거: 1) 2) ... / ① ② ... / 1. 2.
        t = re.sub(r"(?:^|\s)[0-9]{1,2}\)|(?:^|\s)[0-9]{1,2}\.", " ", t)
        t = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", " ", t)

        # 불필요 특수문자 제거 (한글/영문/숫자/공백/일부 의학 기호만 유지)
        t = re.sub(r"[^가-힣A-Za-z0-9\s%+/.-]", " ", t)

        # 공백 정리
        t = re.sub(r"\s+", " ", t).strip()
        return t
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        BM25 인덱스 구축
        Args:
            chunks: [{
                "chunk_id": str,
                "text": str,
                ...
            }, ...]
        """
        self.chunks = chunks
        
        # 텍스트 추출 및 토크나이제이션
        texts = [chunk['text'] for chunk in chunks]
        total = len(texts)
        self.logger.info("BM25 tokenization started: total_chunks=%s", total)
        tokenized_corpus = []
        for i, text in enumerate(tqdm(texts, desc="BM25 tokenizing", total=total), start=1):
            tokenized_corpus.append(self._tokenize(text))
            if i % 5000 == 0 or i == total:
                self.logger.info("BM25 tokenization progress: %s/%s chunks", i, total)
        
        # BM25 인덱스 생성
        self.bm25 = BM25Okapi(tokenized_corpus)
        avg_token_len = float(np.mean([len(toks) for toks in tokenized_corpus])) if tokenized_corpus else 0.0
        self.logger.info("BM25 index built with %s chunks (avg_tokens=%.2f)", len(chunks), avg_token_len)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        BM25 검색
        Args:
            query: 검색 쿼리
            top_k: 상위 k개 결과
        
        Returns:
            [{
                "chunk_id": str,
                "text": str,
                "score": float,
                "rank": int
            }, ...]
        """
        if self.bm25 is None or self.chunks is None:
            raise RuntimeError("Index not built yet. Call build_index() first.")
        
        # 쿼리 토크나이제이션
        query_tokens = self._tokenize(query)
        self.logger.debug(
            "BM25 query tokenized: query_len=%s token_count=%s sample_tokens=%s",
            len(query),
            len(query_tokens),
            query_tokens[:12],
        )
        
        # BM25 점수 계산
        scores = np.asarray(self.bm25.get_scores(query_tokens))
        
        # Top-k 인덱스 추출
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # 결과 구성
        results = []
        for rank, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(scores[idx]),
                "rank": rank + 1,
                "doc_id": chunk.get("doc_id"),
                "source": chunk.get("source"),
                "retriever": "BM25"
            })

        self.logger.info(
            "BM25 search done: query_len=%s top_k=%s returned=%s best_score=%.4f",
            len(query),
            top_k,
            len(results),
            results[0]["score"] if results else 0.0,
        )
        
        return results
