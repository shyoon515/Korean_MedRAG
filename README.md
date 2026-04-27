# 하이브리드 RAG 성능 분석 시스템

의료 도메인에서 BM25 (Sparse) + Dense Embedding (Dense) 기반 하이브리드 검색의 최적 결합 전략을 연구합니다.

## 주요 기능

### 1. 데이터 관리
- **한국어 Corpus만 로드** (영문 제외, lexical fairness)
- Recursive 청킹 (chunk_size=1000, overlap=100)
- Train + Test QA 합산 후 파일럿 set 자동 생성

### 2. 검색 방법
- **BM25**: 형태소 기반 sparse 검색 (kiwipiepy)
- **Dense**: FAISS 기반 embedding 검색 (SentenceTransformer)
- **Hybrid-RRF**: Reciprocal Rank Fusion 결합
- **Hybrid-Alpha**: 가중치 기반 선형 결합 (α=0.0~1.0)

### 3. 평가 지표
- **Retrieval**: 검색 결과의 관련성 점수
- **Generation**: F1 점수 + LLM-as-a-judge (선택)

### 4. 최적화 파이프라인
```
Pilot Set (50 QA)
    ↓
α 값 변화에 따른 성능 측정 (0.0, 0.1, ..., 1.0)
    ↓
최적 α 선정 (Retrieval + Generation 결합 점수 최대)
    ↓
Remaining Set에 최적 α 적용 및 최종 평가
    ↓
결과 분석 (성능 향상도 계산)
```

## 파일 구조

```
src/
├── core/
│   ├── dataset/
│   │   ├── corpus_loader.py    # 한국어 corpus 로드
│   │   └── qa_loader.py        # QA 데이터 + 파일럿 set 생성
│   ├── retrievers/
│   │   ├── base.py             # BaseRetriever 인터페이스
│   │   ├── bm25.py             # BM25Retriever
│   │   ├── dense.py            # DenseRetriever (FAISS)
│   │   └── hybrid.py           # HybridRetriever (RRF + Alpha)
│   ├── evaluators/
│   │   ├── metrics.py          # 기본 평가 지표 (F1, NDCG 등)
│   │   └── llm_evaluator.py    # LLM 기반 평가
│   ├── generator/
│   ├── chain/
│   └── utils/
├── build_sparse_cache.py
└── build_kormedmcqa_sparse_cache.py
```

## 사용 방법

### 1. 환경 설정

필요 패키지 설치:
```bash
pip install rank-bm25 kiwipiepy sentence-transformers faiss-cpu tqdm langchain-text-splitters
```

또는 conda:
```bash
conda install -c conda-forge faiss-cpu
pip install rank-bm25 kiwipiepy sentence-transformers tqdm langchain-text-splitters
```

### 2. 실험 실행

```bash
python src/build_sparse_cache.py --top-k 20
python src/build_kormedmcqa_sparse_cache.py --top-k 20
python src/build_dense_cache.py --encoder BAAI/bge-m3
python src/build_kormedmcqa_dense_cache.py --encoder BAAI/bge-m3
```

### 3. 커스텀 설정

현재 구조에서는 `src` 모듈을 직접 임포트해서 사용:
```python
from src.core import RAGChain

chain = RAGChain(
  retrieval_mode="sparse_only",
  generator_type="openai",
  generator_name="gpt-4o-mini",
  sparse_cache_path="retrieval_cache/bm25",
)

result = chain.ask("질문")
```

## 예상 결과

### Pilot Set에서의 Alpha 최적화
```
Alpha      Retrieval    Generation F1    Combined
0.0        0.6200       0.5410           0.5805
0.1        0.6350       0.5520           0.5935
0.2        0.6480       0.5630           0.6055
...
0.5        0.6850       0.5950           0.6400  ← 최적
...
1.0        0.6200       0.5410           0.5805
```

### Remaining Set 최종 평가
```
Method                  Retrieval    Generation F1
BM25                    0.6200       0.5410
Dense                   0.6100       0.5320
Hybrid-RRF              0.6450       0.5680
Hybrid-Alpha-0.5        0.6850       0.5950  ← 최적 alpha 적용
```

## 분석 포인트

### 1. 평균 성능 향상
- Hybrid-Alpha vs BM25: +5-10% 개선 예상
- Hybrid-Alpha vs Dense: +3-8% 개선 예상

### 2. 질의 유형별 분석
- 단순 증상 질의: BM25 우수
- 서술형 임상 질의: Dense 우수
- 법규/가이드라인: 혼합

### 3. 상관성 분석
- Retrieval score vs Generation F1의 피어슨 상관계수
- 검색 품질이 생성 품질에 미치는 영향도

## 출력 파일

`outputs/experiment_results_YYYYMMDD_HHMMSS.json`:
```json
{
  "config": {
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "top_k": 5,
    "pilot_size": 50,
    "total_qa": 13756,
    "total_chunks": 45000
  },
  "pilot_set_results": [
    {
      "method": "Hybrid-Alpha-0.0",
      "alpha": 0.0,
      "avg_retrieval_score": 0.62,
      "avg_generation_f1": 0.541,
      "combined_score": 0.5805
    },
    ...
  ],
  "final_results": {
    "best_alpha": 0.5,
    "remaining_set": {
      "BM25": {...},
      "Dense": {...},
      "Hybrid-RRF": {...},
      "Hybrid-Alpha-0.5": {...}
    }
  }
}
```

## 주의사항

1. **첫 실행**: Dense 임베딩 생성에 시간 소요 (수십 분)
2. **메모리**: FAISS 인덱싱에 충분한 GPU/RAM 필요
3. **LLMeval**: LLM 클라이언트 없으면 프롬프트만 생성 (평가 값은 0)

## 학회 발표 자료 생성

결과 분석 후:
- 테이블: Alpha 값별 성능 추이
- 그래프: Retrieval vs Generation 상관성
- 케이스 분석: 최적 alpha가 우수한 예시
