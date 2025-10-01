"""
Chunkers module for splitting text into processable segments.
"""
from .base_chunker import BaseChunker
from .sentence_chunker import SentenceChunker
from .chunker_registry import ChunkerRegistry

# Create and populate the global registry
_registry = ChunkerRegistry()
_registry.register(SentenceChunker())


def get_chunker_registry() -> ChunkerRegistry:
    """Get the global chunker registry."""
    return _registry


__all__ = [
    "BaseChunker",
    "SentenceChunker",
    "ChunkerRegistry",
    "get_chunker_registry",
]