from __future__ import annotations

import json
import logging
from typing import Any, Dict


logger = logging.getLogger("rag_api")


def configure_logging() -> None:
    if logger.handlers:
        return

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    logger.propagate = False


def log_json(payload: Dict[str, Any]) -> None:
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))