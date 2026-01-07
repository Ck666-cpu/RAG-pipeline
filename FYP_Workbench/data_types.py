# FYP_Workbench/data_types.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class SourceNode:
    """Represents a specific chunk of text used by the bot"""
    file_name: str
    content_snippet: str  # The actual text paragraph
    score: float  # The relevance score (0.0 to 1.0)


@dataclass
class CRAGResult:
    answer: str
    # We use this to hold rich debug info (SourceNode objects)
    source_nodes: List[SourceNode] = field(default_factory=list)

    # We keep this for backward compatibility if needed, or simple string lists
    sources: List[str] = field(default_factory=list)

    confidence: float = 0.0