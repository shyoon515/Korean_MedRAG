"""
검색 및 생성 평가 지표
"""
from typing import List, Dict, Any
import re
import string
import numpy as np


class RetrievalEvaluator:
    """검색 평가 지표"""
    
    @staticmethod
    def calculate_recall_at_k(retrieved_chunks: List[Dict[str, Any]], 
                             relevant_chunk_ids: List[str], 
                             k: int = 5) -> float:
        """
        Recall@k 계산
        Args:
            retrieved_chunks: 검색된 청크 (상위 k개)
            relevant_chunk_ids: 관련 청크 ID 리스트
            k: 상위 k개
        
        Returns:
            Recall@k (0~1)
        """
        if not relevant_chunk_ids:
            return 1.0  # 관련 문서가 없으면 recall 1.0
        
        retrieved_ids = [chunk['chunk_id'] for chunk in retrieved_chunks[:k]]
        hits = len(set(retrieved_ids) & set(relevant_chunk_ids))
        
        return hits / len(relevant_chunk_ids)
    
    @staticmethod
    def calculate_mrr_at_k(retrieved_chunks: List[Dict[str, Any]], 
                          relevant_chunk_ids: List[str], 
                          k: int = 5) -> float:
        """
        MRR@k (Mean Reciprocal Rank) 계산
        Args:
            retrieved_chunks: 검색된 청크
            relevant_chunk_ids: 관련 청크 ID 리스트
            k: 상위 k개
        
        Returns:
            MRR@k (0~1)
        """
        retrieved_ids = [chunk['chunk_id'] for chunk in retrieved_chunks[:k]]
        
        for rank, chunk_id in enumerate(retrieved_ids, 1):
            if chunk_id in relevant_chunk_ids:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def calculate_ndcg_at_k(retrieved_chunks: List[Dict[str, Any]], 
                           relevant_chunk_ids: List[str], 
                           k: int = 5) -> float:
        """
        NDCG@k (Normalized Discounted Cumulative Gain) 계산
        Args:
            retrieved_chunks: 검색된 청크
            relevant_chunk_ids: 관련 청크 ID 리스트
            k: 상위 k개
        
        Returns:
            NDCG@k (0~1)
        """
        # DCG 계산
        dcg = 0.0
        retrieved_ids = [chunk['chunk_id'] for chunk in retrieved_chunks[:k]]
        for rank, chunk_id in enumerate(retrieved_ids, 1):
            if chunk_id in relevant_chunk_ids:
                dcg += 1.0 / (1 + np.log2(rank))
        
        # IDCG 계산 (최적 순서)
        idcg = 0.0
        for rank in range(1, min(len(relevant_chunk_ids), k) + 1):
            idcg += 1.0 / (1 + np.log2(rank))
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg


class GenerationEvaluator:
    """생성 평가 지표"""
    
    @staticmethod
    def calculate_f1(predicted: str, reference: str) -> float:
        """
        F1 점수 계산 (단어 기반)
        Args:
            predicted: 생성된 답변
            reference: 참고 답변
        
        Returns:
            F1 점수 (0~1)
        """
        # 간단한 토크나이제이션
        pred_tokens = set(GenerationEvaluator._preprocess(predicted).split())
        ref_tokens = set(GenerationEvaluator._preprocess(reference).split())
        
        if not pred_tokens and not ref_tokens:
            return 1.0
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        common = pred_tokens & ref_tokens
        
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    @staticmethod
    def _preprocess(text: str) -> str:
        """텍스트 전처리"""
        # 소문자 변환
        text = text.lower()
        # 특수문자 제거
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    @staticmethod
    def calculate_exact_match(predicted: str, reference: str) -> float:
        """
        Exact Match 계산
        Args:
            predicted: 생성된 답변
            reference: 참고 답변
        
        Returns:
            EM (0 또는 1)
        """
        pred_clean = GenerationEvaluator._preprocess(predicted)
        ref_clean = GenerationEvaluator._preprocess(reference)
        
        return 1.0 if pred_clean == ref_clean else 0.0
    
    @staticmethod
    def calculate_rouge_l(predicted: str, reference: str) -> float:
        """
        ROUGE-L 계산 (LCS 기반)
        간략화된 구현
        """
        pred_words = GenerationEvaluator._preprocess(predicted).split()
        ref_words = GenerationEvaluator._preprocess(reference).split()
        
        if not pred_words or not ref_words:
            return 0.0
        
        # LCS 길이 계산 (간략화)
        lcs_length = len(set(pred_words) & set(ref_words))
        
        precision = lcs_length / len(pred_words)
        recall = lcs_length / len(ref_words)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
