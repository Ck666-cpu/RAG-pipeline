# FYP_Workbench/data_type.py
from dataclasses import dataclass

# This mimics the class from backend/app/services/crag_service.py
@dataclass
class CRAGResult:
    answer: str
    sources: list[str]
    confidence: float

