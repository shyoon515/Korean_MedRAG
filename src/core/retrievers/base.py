"""
기본 Retriever 인터페이스
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseRetriever(ABC):
    """기본 Retriever 인터페이스"""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        검색 수행
        Returns:
            [{
                "chunk_id": str,
                "text": str,
                "score": float,
                ... (other fields)
            }, ...]
        """
        pass
    
    @abstractmethod
    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        인덱스 구축
        Args:
            chunks: 청크 리스트
        """
        pass
