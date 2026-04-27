"""
유틸리티 함수들
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def save_results(results: Dict[str, Any], output_dir: str, name: str = None):
    """결과 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if name is None:
        name = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # JSON 저장
    json_file = output_path / f"{name}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {json_file}")


def load_results(filepath: str) -> Dict[str, Any]:
    """결과 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_chunks(chunks: List[Dict[str, Any]], output_dir: str, name: str = "chunks.pkl"):
    """청크 저장 (pickle)"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pkl_file = output_path / name
    with open(pkl_file, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"Chunks saved to {pkl_file}")


def load_chunks(filepath: str) -> List[Dict[str, Any]]:
    """청크 로드 (pickle)"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def format_results_for_display(results: Dict[str, Any]) -> str:
    """결과를 읽기 좋은 형식으로 변환"""
    output = []
    
    for method, scores in results.items():
        output.append(f"\n{'='*60}")
        output.append(f"Method: {method}")
        output.append(f"{'='*60}")
        
        for metric, value in scores.items():
            if isinstance(value, (int, float)):
                output.append(f"{metric:.<40} {value:.4f}")
            else:
                output.append(f"{metric}: {value}")
    
    return "\n".join(output)


def create_summary_table(results_list: List[Dict[str, Any]]) -> str:
    """여러 결과를 테이블 형식으로 요약"""
    if not results_list:
        return "No results to summarize"
    
    # 모든 메트릭 수집
    all_metrics = set()
    for result in results_list:
        all_metrics.update(result.keys())
    
    all_metrics = sorted(all_metrics)
    
    # 테이블 생성
    lines = []
    
    # 헤더
    header = "Method".ljust(20)
    for metric in all_metrics:
        header += metric[:15].ljust(18)
    lines.append(header)
    lines.append("-" * len(header))
    
    # 데이터 행
    for i, result in enumerate(results_list):
        row = f"Method_{i}".ljust(20)
        for metric in all_metrics:
            value = result.get(metric, 0.0)
            if isinstance(value, float):
                row += f"{value:.4f}".ljust(18)
            else:
                row += str(value)[:15].ljust(18)
        lines.append(row)
    
    return "\n".join(lines)
