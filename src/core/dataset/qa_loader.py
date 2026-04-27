"""
QA 데이터셋 로더 및 파일럿 set 선정
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
import logging


class QALoader:
    """QA 데이터셋 로더"""
    
    def __init__(self, train_qa_dir: str, test_qa_dir: str, logger: logging.Logger = None):
        """
        Args:
            train_qa_dir: train QA 디렉토리
            test_qa_dir: test QA 디렉토리
        """
        self.train_qa_dir = Path(train_qa_dir)
        self.test_qa_dir = Path(test_qa_dir)
        self.logger = logger or logging.getLogger(__name__)
    
    def load_qa_from_folder(self, folder_path: Path) -> List[Dict[str, Any]]:
        """폴더에서 모든 QA 로드"""
        qa_list = []
        
        if not folder_path.exists():
            self.logger.warning("QA folder not found: %s", folder_path)
            return qa_list
        
        json_files = sorted(folder_path.glob("*.json"))
        
        for json_file in json_files:
            try:
                qa = self._load_json_with_fallback(json_file)
                qa_list.append(qa)
            except Exception as e:
                self.logger.exception("Error loading QA file: %s", json_file)
                continue
        
        return qa_list
    
    def load_all_qa(self, split: str = "train") -> List[Dict[str, Any]]:
        """
        모든 QA 로드 (모든 카테고리)
        Args:
            split: "train" 또는 "test"
        """
        base_dir = self.train_qa_dir if split == "train" else self.test_qa_dir
        all_qa = []
        
        # 모든 카테고리 폴더 순회
        category_folders = sorted([d for d in base_dir.iterdir() if d.is_dir()])
        
        for category_folder in category_folders:
            self.logger.info("Loading QA category (%s): %s", split, category_folder.name)
            qa_list = self.load_qa_from_folder(category_folder)
            all_qa.extend(qa_list)
        
        self.logger.info("Total QA loaded (%s): %s", split, len(all_qa))
        return all_qa
    
    def create_pilot_set(self, all_qa: List[Dict[str, Any]], pilot_size: int = 50, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        파일럿 set 생성 (train+test 합쳐서 선정)
        Args:
            all_qa: 모든 QA 리스트
            pilot_size: 파일럿 set 크기
            seed: 재현성 seed
        
        Returns:
            (pilot_set, remaining_set)
        """
        random.seed(seed)
        
        if pilot_size > len(all_qa):
            self.logger.warning("pilot_size %s > total QA %s", pilot_size, len(all_qa))
            pilot_size = len(all_qa)
        
        pilot_indices = set(random.sample(range(len(all_qa)), pilot_size))
        
        pilot_set = []
        remaining_set = []
        
        for idx, qa in enumerate(all_qa):
            if idx in pilot_indices:
                pilot_set.append(qa)
            else:
                remaining_set.append(qa)
        
        self.logger.info("Pilot set size: %s", len(pilot_set))
        self.logger.info("Remaining set size: %s", len(remaining_set))
        
        return pilot_set, remaining_set
    
    def add_metadata(self, qa_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """QA에 메타데이터 추가 (인덱싱용)"""
        for idx, qa in enumerate(qa_list):
            qa['qa_index'] = idx
        return qa_list

    def _load_json_with_fallback(self, json_file: Path) -> Dict[str, Any]:
        """Load JSON robustly for mixed UTF-8 / UTF-8-BOM files."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            with open(json_file, 'r', encoding='utf-8-sig') as f:
                return json.load(f)
