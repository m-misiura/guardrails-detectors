"""
Base class for all chunkers.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[Tuple[str, int, int]]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            **kwargs: Additional chunker-specific parameters
            
        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name/identifier of this chunker."""
        pass