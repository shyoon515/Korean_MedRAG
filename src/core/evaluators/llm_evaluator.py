"""
LLM 기반 평가 (LLMeval retrieval, LLM-as-a-judge generation)
"""
from typing import List, Dict, Any, Optional, Tuple
import json
import re


class LLMEvaluator:
    """LLM 기반 평가"""
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM 클라이언트 (OpenAI, vLLM 등)
                       없으면 프롬프트만 생성
        """
        self.llm_client = llm_client
    
    def evaluate_retrieval_relevance(self, 
                                     query: str, 
                                     retrieved_chunks: List[Dict[str, Any]], 
                                     top_k: int = 5) -> Dict[str, Any]:
        """
        검색 결과의 관련성을 LLM으로 평가
        Args:
            query: 검색 쿼리
            retrieved_chunks: 검색된 청크들
            top_k: 평가할 상위 k개
        
        Returns:
            {
                "query": str,
                "relevance_scores": [float, ...],  # 각 청크별 관련성 점수 (0-10)
                "avg_relevance": float,
                "eval_details": str or None
            }
        """
        # 상위 k개만 사용
        chunks_to_eval = retrieved_chunks[:top_k]
        
        # 평가 프롬프트 생성
        prompt = self._generate_retrieval_eval_prompt(query, chunks_to_eval)
        
        if self.llm_client is None:
            # LLM 없이 프롬프트만 반환
            return {
                "query": query,
                "prompt": prompt,
                "relevance_scores": [0.0] * len(chunks_to_eval),
                "avg_relevance": 0.0,
                "error": "No LLM client provided"
            }
        
        try:
            # LLM으로 평가 수행
            response = self.llm_client.generate(prompt)
            scores = self._parse_retrieval_scores(response)
            
            return {
                "query": query,
                "relevance_scores": scores,
                "avg_relevance": sum(scores) / len(scores) if scores else 0.0,
                "eval_details": response
            }
        except Exception as e:
            return {
                "query": query,
                "relevance_scores": [0.0] * len(chunks_to_eval),
                "avg_relevance": 0.0,
                "error": str(e)
            }
    
    def evaluate_generation_quality(self,
                                    question: str,
                                    generated_answer: str,
                                    reference_answer: str,
                                    retrieved_context: str = None) -> Dict[str, Any]:
        """
        생성 답변의 품질을 LLM으로 평가
        Args:
            question: 질문
            generated_answer: 생성된 답변
            reference_answer: 참고 답변
            retrieved_context: 검색된 문맥 (선택)
        
        Returns:
            {
                "accuracy_score": float (0-10),
                "faithfulness_score": float (0-10),
                "completeness_score": float (0-10),
                "avg_score": float (0-10),
                "eval_details": str
            }
        """
        prompt = self._generate_generation_eval_prompt(
            question, generated_answer, reference_answer, retrieved_context
        )
        
        if self.llm_client is None:
            return {
                "prompt": prompt,
                "accuracy_score": 0.0,
                "faithfulness_score": 0.0,
                "completeness_score": 0.0,
                "avg_score": 0.0,
                "error": "No LLM client provided"
            }
        
        try:
            response = self.llm_client.generate(prompt)
            scores = self._parse_generation_scores(response)
            
            return {
                "accuracy_score": scores.get('accuracy', 0.0),
                "faithfulness_score": scores.get('faithfulness', 0.0),
                "completeness_score": scores.get('completeness', 0.0),
                "avg_score": sum(scores.values()) / len(scores) if scores else 0.0,
                "eval_details": response
            }
        except Exception as e:
            return {
                "accuracy_score": 0.0,
                "faithfulness_score": 0.0,
                "completeness_score": 0.0,
                "avg_score": 0.0,
                "error": str(e)
            }
    
    def _generate_retrieval_eval_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """검색 관련성 평가 프롬프트 생성"""
        chunks_text = "\n\n".join([
            f"[문서 {i+1}]\n{chunk['text'][:500]}" 
            for i, chunk in enumerate(chunks)
        ])
        
        prompt = f"""당신은 의료 정보 검색 시스템의 평가자입니다.
다음 질문에 대해 검색된 문서들이 얼마나 관련성이 있는지 평가하세요.

[질문]
{query}

[검색된 문서들]
{chunks_text}

각 문서에 대해 0-10 점수로 평가하세요. (0=무관련, 10=매우 관련)
응답 형식: [점수1, 점수2, 점수3, ...]

평가 점수를 JSON 형식으로 제공하세요:
{{"scores": [점수1, 점수2, ...]}}
"""
        return prompt
    
    def _generate_generation_eval_prompt(self, 
                                        question: str,
                                        generated_answer: str,
                                        reference_answer: str,
                                        context: str = None) -> str:
        """생성 품질 평가 프롬프트 생성"""
        context_section = f"\n[검색된 문맥]\n{context[:1000]}" if context else ""
        
        prompt = f"""당신은 의료 QA 시스템의 평가자입니다.
생성된 답변의 품질을 다음 관점에서 평가하세요:

1. 정확성 (Accuracy): 의료 정보로서의 정확성
2. 충실성 (Faithfulness): 제공된 문맥과의 일치도
3. 완전성 (Completeness): 질문에 대한 완전한 답변 제공

[질문]
{question}

[생성된 답변]
{generated_answer}

[참고 답변]
{reference_answer}
{context_section}

각 항목을 0-10점으로 평가하세요.
응답 형식 (JSON):
{{
    "accuracy": 점수 (0-10),
    "faithfulness": 점수 (0-10),
    "completeness": 점수 (0-10),
    "explanation": "간단한 설명"
}}
"""
        return prompt
    
    def _parse_retrieval_scores(self, response: str) -> List[float]:
        """검색 평가 응답 파싱"""
        try:
            # JSON 추출 시도
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('scores', [])
            
            # 숫자 배열 추출 시도
            numbers = re.findall(r'\d+(?:\.\d+)?', response)
            return [float(n) / 10.0 if float(n) > 1 else float(n) for n in numbers]
        except:
            return []
    
    def _parse_generation_scores(self, response: str) -> Dict[str, float]:
        """생성 평가 응답 파싱"""
        try:
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    'accuracy': data.get('accuracy', 0.0) / 10.0,
                    'faithfulness': data.get('faithfulness', 0.0) / 10.0,
                    'completeness': data.get('completeness', 0.0) / 10.0
                }
        except:
            pass
        
        return {
            'accuracy': 0.0,
            'faithfulness': 0.0,
            'completeness': 0.0
        }
