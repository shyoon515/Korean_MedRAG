"""
한국어 corpus만 로드하고 청킹하는 모듈
영문 corpus는 제외
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging


class CorpusLoader:
    """한국어 corpus만 로드하고 청킹"""
    
    def __init__(
        self,
        corpus_dir: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        max_docs_per_folder: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            corpus_dir: corpus 루트 디렉토리
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
        """
        self.corpus_dir = Path(corpus_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_docs_per_folder = max_docs_per_folder
        self.logger = logger or logging.getLogger(__name__)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                "다. ",
                "요. ",
                ". ",
                "!",
                "?",
            ],
            length_function=len,
            keep_separator="end"
        )
    
    def load_korean_corpus(self) -> List[Dict[str, Any]]:
        """
        한국어 corpus만 로드
        Returns: 
            [{
                "chunk_id": str (unique),
                "text": str,
                "doc_id": str (원본 문서 ID),
                "source": str (폴더명)
            }, ...]
        """
        chunks = []
        total_docs = 0
        error_count = 0
        start_time = time.time()
        
        korean_folders = self._discover_korean_corpus_folders()
        self.logger.info("Discovered %s TS_국문* corpus folders", len(korean_folders))

        for folder_name in korean_folders:
            folder_path = self.corpus_dir / folder_name
            if not folder_path.exists():
                self.logger.warning("Corpus folder not found: %s", folder_name)
                continue
            
            self.logger.info("Loading corpus folder: %s", folder_name)
            json_files = sorted(folder_path.glob("*.json"))
            if self.max_docs_per_folder is not None:
                json_files = json_files[: self.max_docs_per_folder]
                self.logger.info(
                    "Applying max_docs_per_folder=%s for %s",
                    self.max_docs_per_folder,
                    folder_name,
                )
            
            folder_doc_count = 0
            folder_chunk_count = 0
            for json_file in tqdm(json_files, desc=folder_name):
                try:
                    doc = self._load_json_with_fallback(json_file)
                    
                    if 'content' not in doc:
                        continue
                    
                    doc_id = doc.get('c_id', json_file.stem)
                    content = doc['content']
                    
                    # 청킹
                    text_chunks = self.text_splitter.split_text(content)
                    folder_doc_count += 1
                    total_docs += 1
                    
                    for chunk_idx, chunk_text in enumerate(text_chunks):
                        chunks.append({
                            "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                            "text": chunk_text,
                            "doc_id": doc_id,
                            "source": folder_name
                        })
                        folder_chunk_count += 1
                
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        self.logger.warning("Error loading corpus file: %s (%s)", json_file, e)
                    continue

            self.logger.info(
                "Loaded folder=%s docs=%s chunks=%s",
                folder_name,
                folder_doc_count,
                folder_chunk_count,
            )
        
        elapsed = time.time() - start_time
        self.logger.info(
            "Corpus loading completed: docs=%s chunks=%s errors=%s elapsed=%.2fs",
            total_docs,
            len(chunks),
            error_count,
            elapsed,
        )
        return chunks

    def _discover_korean_corpus_folders(self) -> List[str]:
        """Discover all Korean corpus folders matching TS_국문* under corpus root."""
        if not self.corpus_dir.exists():
            self.logger.warning("Corpus root not found: %s", self.corpus_dir)
            return []

        folders = sorted(
            [
                p.name
                for p in self.corpus_dir.iterdir()
                if p.is_dir() and p.name.startswith("TS_국문")
            ]
        )
        return folders

    def _load_json_with_fallback(self, json_file: Path) -> Dict[str, Any]:
        """Load JSON robustly for mixed UTF-8 / UTF-8-BOM files."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            with open(json_file, 'r', encoding='utf-8-sig') as f:
                return json.load(f)
    
    def get_chunk_by_id(self, chunks: List[Dict[str, Any]], chunk_id: str) -> Dict[str, Any] | None:
        """청크 ID로 청크 조회"""
        for chunk in chunks:
            if chunk['chunk_id'] == chunk_id:
                return chunk
        return None
