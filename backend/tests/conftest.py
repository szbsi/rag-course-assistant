"""
Shared pytest configuration.
Heavy dependencies (chromadb, sentence_transformers) are replaced with MagicMock
at the sys.modules level so backend modules can be imported without real initialization.
"""
import sys
import os
from unittest.mock import MagicMock

# Ensure backend/ is on the path for all test files
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Patch heavy deps at import time (module-level, before any test file imports backend code)
_HEAVY_MOCKS = {
    "chromadb": MagicMock(),
    "chromadb.config": MagicMock(),
    "chromadb.utils": MagicMock(),
    "chromadb.utils.embedding_functions": MagicMock(),
    "sentence_transformers": MagicMock(),
}
sys.modules.update(_HEAVY_MOCKS)
