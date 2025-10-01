"""
Registry for managing available chunkers.
"""
from typing import Dict, List, Optional
from .base_chunker import BaseChunker


class ChunkerRegistry:
    """Registry for managing and accessing chunkers."""
    
    def __init__(self):
        self._chunkers: Dict[str, BaseChunker] = {}
    
    def register(self, chunker: BaseChunker) -> None:
        """Register a chunker."""
        self._chunkers[chunker.name] = chunker
    
    def get(self, name: str) -> Optional[BaseChunker]:
        """Get a chunker by name. Returns None if not found."""
        return self._chunkers.get(name)
    
    def list_names(self) -> List[str]:
        """List all registered chunker names."""
        return list(self._chunkers.keys())