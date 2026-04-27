import logging
from pathlib import Path
from typing import Optional, Dict


def setup_logging(
    logger_name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a console logger and optional file logger with a unified format."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / f"{logger_name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_category_loggers(
    log_dir: str,
    level: int = logging.INFO,
    base_name: str = "sparse_bm25",
) -> Dict[str, logging.Logger]:
    """Create dedicated loggers per category with separate files.

    Returns keys:
      - pipeline: high-level run progress
      - retrieval: retrieval results and top-k details
      - generation: generation evaluation details
      - evaluation: LLM-eval and metric summaries
    """
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    def _build_logger(name_suffix: str, file_name: str, with_console: bool = False) -> logging.Logger:
        logger_name = f"{base_name}.{name_suffix}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = False

        if logger.handlers:
            logger.handlers.clear()

        file_handler = logging.FileHandler(path / file_name, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if with_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    return {
        "pipeline": _build_logger("pipeline", "pipeline_progress.log", with_console=True),
        "retrieval": _build_logger("retrieval", "retrieval_results.log"),
        "generation": _build_logger("generation", "generation_results.log"),
        "evaluation": _build_logger("evaluation", "evaluation_results.log"),
    }
