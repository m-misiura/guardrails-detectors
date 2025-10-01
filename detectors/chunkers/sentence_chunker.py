import re
from typing import List, Tuple
from .base_chunker import BaseChunker


class SentenceChunker(BaseChunker):
    """Chunk text into sentences using regex pattern."""
    DEFAULT_PATTERN = r'[.!?]+(?=\s+[A-Z]|$)'
    @property
    def name(self) -> str:
        return "sentence"
    
    def chunk(self, text: str, pattern: str = None, **kwargs) -> List[Tuple[str, int, int]]:
        """
        Split text into sentences.
        
        Args:
            text: Input text to split
            pattern: Optional custom regex pattern
            **kwargs: Additional parameters (ignored)
            
        Returns:
            List of (sentence, start_pos, end_pos) tuples
        """
        if not text.strip():
            return []
        
        regex = re.compile(pattern or self.DEFAULT_PATTERN)
        sentences = []
        last_end = 0
        for match in regex.finditer(text):
            raw_sentence = text[last_end:match.end()]
            stripped = raw_sentence.strip()
            if stripped:
                leading_spaces = len(raw_sentence) - len(raw_sentence.lstrip())
                start_pos = last_end + leading_spaces
                end_pos = start_pos + len(stripped)
                sentences.append((stripped, start_pos, end_pos))
            last_end = match.end()
        if last_end < len(text):
            raw_sentence = text[last_end:]
            stripped = raw_sentence.strip()
            if stripped:
                leading_spaces = len(raw_sentence) - len(raw_sentence.lstrip())
                start_pos = last_end + leading_spaces
                end_pos = start_pos + len(stripped)
                sentences.append((stripped, start_pos, end_pos))
        return sentences or [(text.strip(), 0, len(text))] if text.strip() else []