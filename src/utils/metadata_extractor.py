"""Comprehensive document metadata extraction for enhanced RAG retrieval.

This module provides automatic metadata extraction from documents during ingestion:
1. Rule-based extraction: Temporal, financial, and entity metadata using regex
2. LLM-based extraction: Semantic metadata like topics, summaries, and key points

The extracted metadata is stored with each chunk to enable:
- Temporal filtering (e.g., "only Q2 2025 results")
- Entity-based filtering (e.g., "only Uber mentions")
- Semantic categorization for improved retrieval
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import logging

from .temporal_utils import extract_temporal_entities, TemporalEntity

logger = logging.getLogger(__name__)


# ============================================================================
# Financial Metadata Extraction (Rule-Based)
# ============================================================================

# Currency patterns
CURRENCY_PATTERNS = [
    # $1.5 billion, $46.8B, €100M
    (r'\$[\d,.]+\s*(?:billion|million|trillion|B|M|T|bn|mn)', 'usd'),
    (r'€[\d,.]+\s*(?:billion|million|trillion|B|M|T|bn|mn)', 'eur'),
    (r'£[\d,.]+\s*(?:billion|million|trillion|B|M|T|bn|mn)', 'gbp'),
    # USD 1.5 billion, EUR 100 million
    (r'(?:USD|EUR|GBP|JPY)\s*[\d,.]+\s*(?:billion|million|trillion)?', 'named'),
]

# Percentage patterns
PERCENTAGE_PATTERN = r'[-+]?\d+(?:\.\d+)?%'

# Financial metrics
FINANCIAL_METRICS = [
    'revenue', 'gross bookings', 'ebitda', 'adjusted ebitda', 'net income',
    'operating income', 'gross profit', 'free cash flow', 'operating cash flow',
    'earnings per share', 'eps', 'gaap', 'non-gaap', 'margin', 'growth',
    'year-over-year', 'yoy', 'quarter-over-quarter', 'qoq'
]


@dataclass
class FinancialMetadata:
    """Extracted financial metadata."""
    currency_mentions: List[str] = field(default_factory=list)
    percentages: List[str] = field(default_factory=list)
    metrics_mentioned: List[str] = field(default_factory=list)
    is_financial_document: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "currency_mentions": self.currency_mentions,
            "percentages": self.percentages,
            "metrics_mentioned": self.metrics_mentioned,
            "is_financial_document": self.is_financial_document,
        }


def extract_financial_metadata(text: str) -> FinancialMetadata:
    """Extract financial information from text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        FinancialMetadata object with extracted information
    """
    if not text:
        return FinancialMetadata()
    
    text_lower = text.lower()
    
    # Extract currency mentions
    currency_mentions = []
    for pattern, _ in CURRENCY_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        currency_mentions.extend(matches)
    
    # Extract percentages
    percentages = re.findall(PERCENTAGE_PATTERN, text)
    
    # Check for financial metrics
    metrics_found = []
    for metric in FINANCIAL_METRICS:
        if metric in text_lower:
            metrics_found.append(metric)
    
    # Determine if this is a financial document
    is_financial = len(metrics_found) >= 2 or len(currency_mentions) >= 2
    
    return FinancialMetadata(
        currency_mentions=list(set(currency_mentions))[:10],  # Limit to 10
        percentages=list(set(percentages))[:10],
        metrics_mentioned=list(set(metrics_found)),
        is_financial_document=is_financial,
    )


# ============================================================================
# Entity Metadata Extraction (Rule-Based)
# ============================================================================

# Common company suffixes
COMPANY_SUFFIXES = r'(?:Inc|Corp|Corporation|Ltd|LLC|Co|Company|Technologies|Tech|Group|Holdings)?\.?'

# Product/service keywords
PRODUCT_KEYWORDS = [
    'uber eats', 'uber freight', 'uber one', 'uber pass',
    # Add more as needed based on your documents
]


@dataclass
class EntityMetadata:
    """Extracted entity metadata."""
    companies: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "companies": self.companies,
            "products": self.products,
            "locations": self.locations,
            "people": self.people,
        }


def extract_entity_metadata(text: str) -> EntityMetadata:
    """Extract named entities from text using pattern matching.
    
    Args:
        text: Input text to analyze
        
    Returns:
        EntityMetadata object with extracted entities
    """
    if not text:
        return EntityMetadata()
    
    text_lower = text.lower()
    
    # Find product mentions
    products_found = []
    for product in PRODUCT_KEYWORDS:
        if product in text_lower:
            products_found.append(product.title())
    
    # Simple pattern for company names (Proper Case followed by company suffix)
    # This is a basic heuristic; for production, consider spaCy NER
    company_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+' + COMPANY_SUFFIXES + r'\b'
    company_matches = re.findall(company_pattern, text)
    
    # Also look for specific known companies
    known_companies = ['Uber', 'Google', 'Apple', 'Microsoft', 'Amazon', 'Meta', 'Tesla']
    companies_found = []
    for company in known_companies:
        if company.lower() in text_lower:
            companies_found.append(company)
    companies_found.extend([m for m in company_matches if m not in companies_found])
    
    return EntityMetadata(
        companies=list(set(companies_found))[:10],
        products=list(set(products_found))[:10],
        locations=[],  # Would need NER for accurate location extraction
        people=[],  # Would need NER for accurate people extraction
    )


# ============================================================================
# Temporal Metadata Wrapper
# ============================================================================

@dataclass
class TemporalMetadata:
    """Structured temporal metadata."""
    quarters: List[str] = field(default_factory=list)
    years: List[int] = field(default_factory=list)
    months: List[str] = field(default_factory=list)
    fiscal_years: List[str] = field(default_factory=list)
    raw_entities: List[TemporalEntity] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "quarters": self.quarters,
            "years": self.years,
            "months": self.months,
            "fiscal_years": self.fiscal_years,
        }


def extract_temporal_metadata(text: str) -> TemporalMetadata:
    """Extract temporal metadata using the temporal_utils module.
    
    Args:
        text: Input text to analyze
        
    Returns:
        TemporalMetadata object with structured temporal information
    """
    entities = extract_temporal_entities(text)
    
    quarters = []
    years = set()
    months = []
    fiscal_years = []
    
    for entity in entities:
        if entity.entity_type == 'quarter' and entity.year and entity.quarter:
            quarters.append(f"Q{entity.quarter} {entity.year}")
            years.add(entity.year)
        elif entity.entity_type == 'year':
            years.add(entity.year)
        elif entity.entity_type == 'month' and entity.year and entity.month:
            months.append(f"{entity.month}/{entity.year}")
            years.add(entity.year)
        elif entity.entity_type == 'fiscal_year':
            fiscal_years.append(entity.normalized)
            if entity.year:
                years.add(entity.year)
    
    return TemporalMetadata(
        quarters=list(set(quarters)),
        years=sorted(list(years)),
        months=list(set(months)),
        fiscal_years=list(set(fiscal_years)),
        raw_entities=entities,
    )


# ============================================================================
# LLM-Based Semantic Metadata Extraction
# ============================================================================

@dataclass
class SemanticMetadata:
    """LLM-extracted semantic metadata."""
    topics: List[str] = field(default_factory=list)
    summary: str = ""
    document_type: str = "general"
    key_points: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topics": self.topics,
            "summary": self.summary,
            "document_type": self.document_type,
            "key_points": self.key_points,
        }


SEMANTIC_EXTRACTION_PROMPT = """Analyze the following text and extract metadata in JSON format.

Text:
{text}

Extract the following (respond with valid JSON only, no markdown):
{{
    "topics": ["list of 2-3 main topics"],
    "summary": "1-2 sentence summary of the content",
    "document_type": "one of: earnings_report, press_release, technical_doc, legal, general",
    "key_points": ["list of 2-4 key facts or figures mentioned"]
}}"""


def extract_semantic_metadata(
    text: str,
    llm_wrapper=None,
    max_text_length: int = 2000,
) -> SemanticMetadata:
    """Extract semantic metadata using LLM.
    
    Args:
        text: Input text to analyze
        llm_wrapper: LLMWrapper instance for generation
        max_text_length: Maximum text length to send to LLM
        
    Returns:
        SemanticMetadata object with extracted information
    """
    if not text or not llm_wrapper:
        return SemanticMetadata()
    
    try:
        # Truncate text if too long
        truncated_text = text[:max_text_length] if len(text) > max_text_length else text
        
        prompt = SEMANTIC_EXTRACTION_PROMPT.format(text=truncated_text)
        response = llm_wrapper.generate(prompt)
        
        # Parse JSON response
        import json
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return SemanticMetadata(
                topics=data.get("topics", [])[:5],
                summary=data.get("summary", "")[:300],
                document_type=data.get("document_type", "general"),
                key_points=data.get("key_points", [])[:5],
            )
    except Exception as e:
        logger.warning(f"Failed to extract semantic metadata: {e}")
    
    return SemanticMetadata()


# ============================================================================
# Unified Document Metadata Extractor
# ============================================================================

@dataclass
class DocumentMetadata:
    """Complete extracted metadata for a document chunk."""
    temporal: TemporalMetadata = field(default_factory=TemporalMetadata)
    financial: FinancialMetadata = field(default_factory=FinancialMetadata)
    entities: EntityMetadata = field(default_factory=EntityMetadata)
    semantic: SemanticMetadata = field(default_factory=SemanticMetadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for storage."""
        return {
            "temporal": self.temporal.to_dict(),
            "financial": self.financial.to_dict(),
            "entities": self.entities.to_dict(),
            "semantic": self.semantic.to_dict(),
            # Flattened fields for easy filtering
            "quarters": self.temporal.quarters,
            "years": self.temporal.years,
            "is_financial": self.financial.is_financial_document,
            "companies": self.entities.companies,
            "topics": self.semantic.topics,
        }


class DocumentMetadataExtractor:
    """Unified metadata extractor combining rule-based and LLM methods."""
    
    def __init__(
        self,
        llm_wrapper=None,
        enable_llm_extraction: bool = True,
        enable_rule_extraction: bool = True,
    ):
        """Initialize the metadata extractor.
        
        Args:
            llm_wrapper: Optional LLMWrapper for semantic extraction
            enable_llm_extraction: Whether to use LLM for semantic metadata
            enable_rule_extraction: Whether to use rule-based extraction
        """
        self.llm_wrapper = llm_wrapper
        self.enable_llm_extraction = enable_llm_extraction and llm_wrapper is not None
        self.enable_rule_extraction = enable_rule_extraction
    
    def extract(self, text: str, use_llm: bool = True) -> DocumentMetadata:
        """Extract all metadata from text.
        
        Args:
            text: Input text to analyze
            use_llm: Whether to use LLM extraction for this call
            
        Returns:
            DocumentMetadata with all extracted information
        """
        metadata = DocumentMetadata()
        
        if self.enable_rule_extraction:
            metadata.temporal = extract_temporal_metadata(text)
            metadata.financial = extract_financial_metadata(text)
            metadata.entities = extract_entity_metadata(text)
        
        if self.enable_llm_extraction and use_llm:
            metadata.semantic = extract_semantic_metadata(text, self.llm_wrapper)
        
        return metadata
    
    def extract_batch(
        self,
        texts: List[str],
        use_llm: bool = False,  # Default off for batch to save time
    ) -> List[DocumentMetadata]:
        """Extract metadata from multiple texts.
        
        Args:
            texts: List of texts to analyze
            use_llm: Whether to use LLM extraction
            
        Returns:
            List of DocumentMetadata objects
        """
        return [self.extract(text, use_llm=use_llm) for text in texts]
