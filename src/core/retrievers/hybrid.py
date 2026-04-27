"""
하이브리드 Retriever (BM25 + Dense)
RRF (Reciprocal Rank Fusion) 및 Alpha Weighted 결합
"""
from typing import List, Dict, Any
from .base import BaseRetriever


class HybridRetriever(BaseRetriever):
    """하이브리드 Retriever"""
    
    def __init__(self, sparse_retriever: BaseRetriever, dense_retriever: BaseRetriever):
        """
        Args:
            sparse_retriever: BM25 Retriever
            dense_retriever: Dense Retriever (FAISS)
        """
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """두 retriever 모두 인덱스 구축"""
        self.sparse_retriever.build_index(chunks)
        self.dense_retriever.build_index(chunks)
    
    def search_rrf(self, query: str, top_k: int = 5, k_rrf: int = 60) -> List[Dict[str, Any]]:
        """
        RRF (Reciprocal Rank Fusion) 기반 검색
        Args:
            query: 검색 쿼리
            top_k: 최종 결과 상위 k개
            k_rrf: RRF 하이퍼파라미터 (보통 60)
        
        Returns:
            [{
                "chunk_id": str,
                "text": str,
                "rrf_score": float,
                "rank": int,
                "in_sparse": bool,
                "in_dense": bool,
                "sparse_rank": int or None,
                "dense_rank": int or None
            }, ...]
        """
        # 두 retriever에서 검색
        sparse_results = self.sparse_retriever.search(query, top_k=top_k)
        dense_results = self.dense_retriever.search(query, top_k=top_k)
        
        # RRF 점수 계산
        rrf_scores = {}
        
        # Sparse 결과
        for result in sparse_results:
            chunk_id = result['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k_rrf + result['rank'])
        
        # Dense 결과
        for result in dense_results:
            chunk_id = result['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k_rrf + result['rank'])
        
        # 결과 구성 (모든 청크 ID를 포함한 맵 생성)
        chunk_info = {}
        for result in sparse_results:
            chunk_id = result['chunk_id']
            chunk_info[chunk_id] = {
                "text": result['text'],
                "sparse_rank": result['rank'],
                "dense_rank": None
            }
        
        for result in dense_results:
            chunk_id = result['chunk_id']
            if chunk_id not in chunk_info:
                chunk_info[chunk_id] = {"text": result['text'], "sparse_rank": None}
            chunk_info[chunk_id]["dense_rank"] = result['rank']
        
        # 정렬
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 최종 결과
        results = []
        for rank, (chunk_id, rrf_score) in enumerate(sorted_results):
            info = chunk_info[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "text": info["text"],
                "rrf_score": float(rrf_score),
                "rank": rank + 1,
                "in_sparse": info["sparse_rank"] is not None,
                "in_dense": info["dense_rank"] is not None,
                "sparse_rank": info["sparse_rank"],
                "dense_rank": info["dense_rank"],
                "method": "RRF"
            })
        
        return results
    
    def search_alpha_weighted(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Alpha Weighted 기반 검색
        Args:
            query: 검색 쿼리
            top_k: 최종 결과 상위 k개
            alpha: Dense 가중치 (0.0~1.0)
                   0.0 = 100% BM25, 1.0 = 100% Dense
        
        Returns:
            [{
                "chunk_id": str,
                "text": str,
                "combined_score": float,
                "rank": int,
                "sparse_score_scaled": float,
                "dense_score_scaled": float,
                "alpha": float
            }, ...]
        """
        assert 0.0 <= alpha <= 1.0, "alpha must be between 0.0 and 1.0"
        
        # 두 retriever에서 검색 (상위 k개보다 많이)
        search_k = top_k * 2  # 더 많은 결과를 가져와서 병합 후 top_k 선택
        sparse_results = self.sparse_retriever.search(query, top_k=search_k)
        dense_results = self.dense_retriever.search(query, top_k=search_k)
        
        # Min-Max 정규화
        sparse_scores = self._minmax_normalize([r['score'] for r in sparse_results])
        dense_scores = self._minmax_normalize([r['score'] for r in dense_results])
        
        # 청크별 점수 딕셔너리
        combined_scores = {}
        chunk_info = {}
        
        # Sparse 결과 처리
        for result, norm_score in zip(sparse_results, sparse_scores):
            chunk_id = result['chunk_id']
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {'sparse': 0, 'dense': 0}
                chunk_info[chunk_id] = {"text": result['text']}
            combined_scores[chunk_id]['sparse'] = norm_score
        
        # Dense 결과 처리
        for result, norm_score in zip(dense_results, dense_scores):
            chunk_id = result['chunk_id']
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {'sparse': 0, 'dense': 0}
                chunk_info[chunk_id] = {"text": result['text']}
            combined_scores[chunk_id]['dense'] = norm_score
        
        # 가중 결합
        final_scores = {}
        for chunk_id, scores in combined_scores.items():
            final_scores[chunk_id] = alpha * scores['dense'] + (1 - alpha) * scores['sparse']
        
        # 정렬
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 최종 결과
        results = []
        for rank, (chunk_id, combined_score) in enumerate(sorted_results):
            results.append({
                "chunk_id": chunk_id,
                "text": chunk_info[chunk_id]["text"],
                "combined_score": float(combined_score),
                "rank": rank + 1,
                "sparse_score_scaled": float(combined_scores[chunk_id]['sparse']),
                "dense_score_scaled": float(combined_scores[chunk_id]['dense']),
                "alpha": alpha,
                "method": f"AlphaWeighted(α={alpha})"
            })
        
        return results
    
    def _minmax_normalize(self, scores: List[float]) -> List[float]:
        """Min-Max 정규화"""
        if not scores:
            return []
        
        scores_array = scores
        min_score = min(scores_array)
        max_score = max(scores_array)
        
        if max_score == min_score:
            return [1.0] * len(scores_array)
        
        return [(s - min_score) / (max_score - min_score) for s in scores_array]
