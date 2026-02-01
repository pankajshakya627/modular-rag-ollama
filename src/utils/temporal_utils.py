"""Temporal entity extraction and matching utilities for time-aware RAG.

This module provides utilities for:
1. Extracting temporal entities (dates, quarters, years) from text
2. Normalizing temporal references to comparable formats
3. Calculating temporal match scores for reranking

Based on research from Temporal RAG best practices:
- Pre-filter or boost documents with matching temporal metadata
- Handle various date formats (Q1 2025, "First Quarter", 2024-01, etc.)
- Apply configurable boost/penalty to reranking scores
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemporalEntity:
    """Represents an extracted temporal entity."""
    raw_text: str
    entity_type: str  # "quarter", "year", "month", "date", "fiscal_year"
    normalized: str  # Normalized form for comparison
    year: Optional[int] = None
    quarter: Optional[int] = None
    month: Optional[int] = None
    
    def __hash__(self):
        return hash(self.normalized)
    
    def __eq__(self, other):
        if not isinstance(other, TemporalEntity):
            return False
        return self.normalized == other.normalized


# Quarter patterns: Q1, Q2, Q3, Q4, 1Q, 2Q, etc.
QUARTER_PATTERNS = [
    # Q1 2025, Q2-2024, Q3/2023
    (r'\b[Qq]([1-4])[\s\-\/]*(20\d{2})\b', 'quarter'),
    # 2025 Q1, 2024-Q2
    (r'\b(20\d{2})[\s\-\/]*[Qq]([1-4])\b', 'quarter_reversed'),
    # 1Q2025, 2Q24
    (r'\b([1-4])[Qq](20\d{2}|\d{2})\b', 'quarter_compact'),
    # First Quarter 2025, Second Quarter 2024
    (r'\b(first|second|third|fourth)\s+quarter\s*(20\d{2})?\b', 'quarter_text'),
    # Q1, Q2 alone (without year)
    (r'\b[Qq]([1-4])\b(?!\s*\d)', 'quarter_no_year'),
]

# Year patterns
YEAR_PATTERNS = [
    (r'\b(20\d{2})\b', 'year'),
    (r'\bFY\s*(20\d{2}|\d{2})\b', 'fiscal_year'),
]

# Month patterns
MONTH_PATTERNS = [
    # January 2025, Feb 2024
    (r'\b(january|february|march|april|may|june|july|august|september|october|november|december|'
     r'jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s*(20\d{2})?\b', 'month'),
    # 2024-01, 2025/03
    (r'\b(20\d{2})[\-\/](0[1-9]|1[0-2])\b', 'year_month'),
]

# Date patterns
DATE_PATTERNS = [
    # 2024-01-15, 2025/03/20
    (r'\b(20\d{2})[\-\/](0[1-9]|1[0-2])[\-\/](0[1-9]|[12]\d|3[01])\b', 'date_iso'),
    # 15 Jan 2024, 20 March 2025
    (r'\b(0?[1-9]|[12]\d|3[01])\s+(january|february|march|april|may|june|july|august|'
     r'september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+'
     r'(20\d{2})\b', 'date_text'),
]

QUARTER_TEXT_MAP = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4
}

MONTH_MAP = {
    'january': 1, 'jan': 1,
    'february': 2, 'feb': 2,
    'march': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'may': 5,
    'june': 6, 'jun': 6,
    'july': 7, 'jul': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10,
    'november': 11, 'nov': 11,
    'december': 12, 'dec': 12,
}


def extract_temporal_entities(text: str) -> List[TemporalEntity]:
    """Extract all temporal entities from text.
    
    Args:
        text: Input text to extract entities from
        
    Returns:
        List of TemporalEntity objects
    """
    if not text:
        return []
    
    entities = []
    text_lower = text.lower()
    
    # Extract quarters
    for pattern, pattern_type in QUARTER_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entity = _parse_quarter_match(match, pattern_type)
            if entity:
                entities.append(entity)
    
    # Extract years (only if not already part of a quarter)
    for pattern, pattern_type in YEAR_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            year_str = match.group(1)
            # Normalize 2-digit years
            if len(year_str) == 2:
                year = 2000 + int(year_str)
            else:
                year = int(year_str)
            
            # Check if this year is part of an existing quarter entity
            is_part_of_quarter = any(
                e.year == year and e.entity_type == 'quarter'
                for e in entities
            )
            if not is_part_of_quarter:
                entities.append(TemporalEntity(
                    raw_text=match.group(0),
                    entity_type=pattern_type,
                    normalized=f"Y{year}",
                    year=year,
                ))
    
    # Extract months
    for pattern, pattern_type in MONTH_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entity = _parse_month_match(match, pattern_type)
            if entity:
                entities.append(entity)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for entity in entities:
        if entity.normalized not in seen:
            seen.add(entity.normalized)
            unique_entities.append(entity)
    
    return unique_entities


def _parse_quarter_match(match: re.Match, pattern_type: str) -> Optional[TemporalEntity]:
    """Parse a quarter regex match into a TemporalEntity."""
    try:
        if pattern_type == 'quarter':
            quarter = int(match.group(1))
            year = int(match.group(2))
        elif pattern_type == 'quarter_reversed':
            year = int(match.group(1))
            quarter = int(match.group(2))
        elif pattern_type == 'quarter_compact':
            quarter = int(match.group(1))
            year_str = match.group(2)
            year = int(year_str) if len(year_str) == 4 else 2000 + int(year_str)
        elif pattern_type == 'quarter_text':
            quarter_word = match.group(1).lower()
            quarter = QUARTER_TEXT_MAP.get(quarter_word, 0)
            year_str = match.group(2)
            year = int(year_str) if year_str else None
        elif pattern_type == 'quarter_no_year':
            quarter = int(match.group(1))
            year = None
        else:
            return None
        
        if year:
            normalized = f"Q{quarter}Y{year}"
        else:
            normalized = f"Q{quarter}"
        
        return TemporalEntity(
            raw_text=match.group(0),
            entity_type='quarter',
            normalized=normalized,
            year=year,
            quarter=quarter,
        )
    except (ValueError, IndexError) as e:
        logger.debug(f"Failed to parse quarter match: {e}")
        return None


def _parse_month_match(match: re.Match, pattern_type: str) -> Optional[TemporalEntity]:
    """Parse a month regex match into a TemporalEntity."""
    try:
        if pattern_type == 'month':
            month_name = match.group(1).lower()
            month = MONTH_MAP.get(month_name, 0)
            year_str = match.group(2)
            year = int(year_str) if year_str else None
        elif pattern_type == 'year_month':
            year = int(match.group(1))
            month = int(match.group(2))
        else:
            return None
        
        if year:
            normalized = f"M{month:02d}Y{year}"
        else:
            normalized = f"M{month:02d}"
        
        return TemporalEntity(
            raw_text=match.group(0),
            entity_type='month',
            normalized=normalized,
            year=year,
            month=month,
        )
    except (ValueError, IndexError) as e:
        logger.debug(f"Failed to parse month match: {e}")
        return None


def calculate_temporal_score(
    query_entities: List[TemporalEntity],
    doc_entities: List[TemporalEntity],
    match_boost: float = 1.5,
    mismatch_penalty: float = 0.7,
    partial_match_boost: float = 1.2,
) -> float:
    """Calculate temporal alignment score between query and document.
    
    Args:
        query_entities: Temporal entities from the query
        doc_entities: Temporal entities from the document
        match_boost: Multiplier for exact matches (e.g., 1.5 = 50% boost)
        mismatch_penalty: Multiplier for mismatches (e.g., 0.7 = 30% penalty)
        partial_match_boost: Multiplier for partial matches (same year, different quarter)
        
    Returns:
        Score multiplier (1.0 = neutral, >1 = boost, <1 = penalty)
    """
    if not query_entities:
        # No temporal constraint in query, neutral score
        return 1.0
    
    if not doc_entities:
        # Query has temporal constraint but doc has no temporal info
        # Slight penalty (document might not be relevant)
        return 0.9
    
    # Build sets for comparison
    query_quarters = {e for e in query_entities if e.entity_type == 'quarter' and e.year}
    query_years = {e.year for e in query_entities if e.year}
    
    doc_quarters = {e for e in doc_entities if e.entity_type == 'quarter' and e.year}
    doc_years = {e.year for e in doc_entities if e.year}
    
    # Check for exact quarter match
    if query_quarters and doc_quarters:
        exact_matches = query_quarters & doc_quarters
        if exact_matches:
            # Exact quarter match - strong boost
            return match_boost
        
        # Check for year match but quarter mismatch
        query_quarter_years = {e.year for e in query_quarters}
        doc_quarter_years = {e.year for e in doc_quarters}
        
        if query_quarter_years & doc_quarter_years:
            # Same year but different quarter
            # Check if the document quarter is mentioned in query
            query_q_nums = {(e.year, e.quarter) for e in query_quarters}
            doc_q_nums = {(e.year, e.quarter) for e in doc_quarters}
            
            if not query_q_nums & doc_q_nums:
                # Same year, different quarter - penalty
                return mismatch_penalty
        else:
            # Different years entirely - stronger penalty
            return mismatch_penalty * 0.9
    
    # Check for year-only match
    if query_years and doc_years:
        if query_years & doc_years:
            # Year match without specific quarter constraint
            return partial_match_boost
        else:
            # Different years
            return mismatch_penalty
    
    # No clear temporal relationship found
    return 1.0


def get_temporal_context(entities: List[TemporalEntity]) -> str:
    """Generate a human-readable temporal context string.
    
    Args:
        entities: List of temporal entities
        
    Returns:
        Human-readable string describing the temporal context
    """
    if not entities:
        return "No specific time period"
    
    parts = []
    for entity in entities:
        if entity.entity_type == 'quarter' and entity.year:
            parts.append(f"Q{entity.quarter} {entity.year}")
        elif entity.entity_type == 'year':
            parts.append(str(entity.year))
        elif entity.entity_type == 'month' and entity.year:
            parts.append(f"{entity.month}/{entity.year}")
    
    return ", ".join(parts) if parts else "No specific time period"
