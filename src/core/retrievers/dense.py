"""
Dense 임베딩 기반 Retriever (FAISS 사용)
"""
import numpy as np
from typing import List, Dict, Any
import faiss

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence_transformers not installed")
    SentenceTransformer = None

from .base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Dense 임베딩 기반 Retriever (FAISS)"""
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        """
        Args:
            model_name: SentenceTransformer 모델명
        """
        self.model_name = model_name
        self.model = None
        self.chunks = None
        self.index = None
        self.embeddings = None
        self._init_model()
    
    def _init_model(self):
        """모델 초기화"""
        if SentenceTransformer is None:
            raise RuntimeError("sentence_transformers not installed")
        
        print(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"Model loaded on device: {self.model.device}")
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Dense 인덱스 구축 (FAISS)
        Args:
            chunks: [{
                "chunk_id": str,
                "text": str,
                ...
            }, ...]
        """
        self.chunks = chunks
        
        # 텍스트 추출
        texts = [chunk['text'] for chunk in chunks]
        
        # 임베딩 생성
        print(f"Generating embeddings for {len(texts)} chunks...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # FAISS 인덱스 생성
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.index.add(np.ascontiguousarray(self.embeddings).astype('float32'))
        
        print(f"Dense index built with {len(chunks)} chunks (dim={dimension})")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Dense 검색
        Args:
            query: 검색 쿼리
            top_k: 상위 k개 결과
        
        Returns:
            [{
                "chunk_id": str,
                "text": str,
                "score": float (similarity, 낮을수록 좋음 - L2 distance),
                "rank": int
            }, ...]
        """
        if self.index is None or self.chunks is None:
            raise RuntimeError("Index not built yet. Call build_index() first.")
        
        # 쿼리 임베딩
        query_embedding = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True).astype('float32')
        
        # FAISS 검색 (L2 distance)
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 거리를 유사도로 변환 (1 / (1 + distance))
        distances = distances[0]  # 배치 크기 1이므로 첫 원소만
        
        # 결과 구성
        results = []
        for rank, idx in enumerate(indices[0]):
            chunk = self.chunks[idx]
            distance = distances[rank]
            # L2 distance를 유사도로 변환 (낮을수록 높은 유사도)
            similarity = 1.0 / (1.0 + distance)
            
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(similarity),  # 높을수록 좋음
                "distance": float(distance),  # L2 distance (낮을수록 좋음)
                "rank": rank + 1,
                "retriever": "Dense"
            })
        
        return results
