"""Document Quality Checker for validating document extraction quality.

This module provides document quality assessment functionality:
- Validates character density and readability of extracted text
- Detects garbled text, scan-only documents, or encoding issues
- Provides configurable thresholds for quality gates

Design Principles:
- Config-Driven: Thresholds configurable via settings
- Fast: Only checks first N pages for efficiency
- Actionable: Provides clear pass/fail with detailed metrics
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from src.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for a document.
    
    Attributes:
        total_chars: Total character count in sample
        valid_chars: Count of valid/printable characters
        valid_ratio: Ratio of valid to total characters (0.0-1.0)
        chinese_chars: Count of Chinese characters
        english_chars: Count of English characters
        digit_chars: Count of digit characters
        space_chars: Count of whitespace characters
        garbage_indicators: List of detected garbage patterns
        has_text_layer: Whether document has extractable text
        page_count: Total pages in document
        sampled_pages: Number of pages sampled for analysis
    """
    total_chars: int
    valid_chars: int
    valid_ratio: float
    chinese_chars: int
    english_chars: int
    digit_chars: int
    space_chars: int
    garbage_indicators: List[str]
    has_text_layer: bool
    page_count: int
    sampled_pages: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_chars": self.total_chars,
            "valid_chars": self.valid_chars,
            "valid_ratio": self.valid_ratio,
            "chinese_chars": self.chinese_chars,
            "english_chars": self.english_chars,
            "digit_chars": self.digit_chars,
            "space_chars": self.space_chars,
            "garbage_indicators": self.garbage_indicators,
            "has_text_layer": self.has_text_layer,
            "page_count": self.page_count,
            "sampled_pages": self.sampled_pages,
        }


@dataclass
class QualityCheckResult:
    """Result of document quality check.
    
    Attributes:
        passed: Whether the document passed quality check
        score: Overall quality score (0.0-1.0)
        metrics: Detailed quality metrics
        reasons: List of failure reasons if failed
    """
    passed: bool
    score: float
    metrics: QualityMetrics
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "metrics": self.metrics.to_dict(),
            "reasons": self.reasons,
        }


class DocumentQualityChecker:
    """Quality checker for validating document extraction results.
    
    This checker analyzes the first few pages of a document to determine
    if the document has sufficient readable text content. It detects:
    - Garbled/corrupted text
    - Scan-only documents (no text layer)
    - Severe encoding issues
    - Too much noise or invalid characters
    
    Attributes:
        min_valid_ratio: Minimum ratio of valid characters (default: 0.8)
        sample_pages: Number of pages to sample for quality check (default: 3)
        min_chars: Minimum character count required (default: 100)
        check_text_layer: Whether to check for text layer existence (default: True)
    
    Example:
        >>> checker = DocumentQualityChecker(min_valid_ratio=0.8, sample_pages=3)
        >>> result = checker.check("documents/report.pdf")
        >>> if not result.passed:
        ...     print(f"Document rejected: {result.reasons}")
    """
    
    def __init__(
        self,
        min_valid_ratio: float = 0.8,
        sample_pages: int = 3,
        min_chars: int = 100,
        check_text_layer: bool = True,
    ):
        """Initialize DocumentQualityChecker.
        
        Args:
            min_valid_ratio: Minimum valid character ratio threshold (0.0-1.0)
            sample_pages: Number of pages to sample for analysis
            min_chars: Minimum characters required in sample
            check_text_layer: Whether to verify text layer exists
        """
        if not 0.0 <= min_valid_ratio <= 1.0:
            raise ValueError("min_valid_ratio must be between 0.0 and 1.0")
        if sample_pages < 1:
            raise ValueError("sample_pages must be at least 1")
        
        self.min_valid_ratio = min_valid_ratio
        self.sample_pages = sample_pages
        self.min_chars = min_chars
        self.check_text_layer = check_text_layer
        
        logger.info(
            f"DocumentQualityChecker initialized: "
            f"min_valid_ratio={min_valid_ratio}, "
            f"sample_pages={sample_pages}, "
            f"min_chars={min_chars}"
        )
    
    def check(self, file_path: str | Path) -> QualityCheckResult:
        """Check document quality.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            QualityCheckResult with pass/fail status and detailed metrics.
        """
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available, skipping quality check")
            return QualityCheckResult(
                passed=True,
                score=1.0,
                metrics=QualityMetrics(
                    total_chars=0, valid_chars=0, valid_ratio=1.0,
                    chinese_chars=0, english_chars=0, digit_chars=0,
                    space_chars=0, garbage_indicators=[],
                    has_text_layer=True, page_count=0, sampled_pages=0
                ),
                reasons=[]
            )
        
        path = Path(file_path)
        if not path.exists():
            return QualityCheckResult(
                passed=False,
                score=0.0,
                metrics=QualityMetrics(
                    total_chars=0, valid_chars=0, valid_ratio=0.0,
                    chinese_chars=0, english_chars=0, digit_chars=0,
                    space_chars=0, garbage_indicators=["file_not_found"],
                    has_text_layer=False, page_count=0, sampled_pages=0
                ),
                reasons=[f"File not found: {file_path}"]
            )
        
        try:
            doc = fitz.open(str(path))
            page_count = len(doc)
            sample_pages = min(self.sample_pages, page_count)
            
            # Collect text from sample pages
            all_text = []
            has_text_layer = False
            
            for page_num in range(sample_pages):
                page = doc[page_num]
                text = page.get_text()
                
                if text and text.strip():
                    has_text_layer = True
                    all_text.append(text)
            
            doc.close()
            
            combined_text = "\n".join(all_text)
            metrics = self._analyze_text(combined_text, has_text_layer, page_count, sample_pages)
            
            # Evaluate based on metrics
            return self._evaluate(metrics)
            
        except Exception as e:
            logger.error(f"Quality check failed for {file_path}: {e}")
            return QualityCheckResult(
                passed=False,
                score=0.0,
                metrics=QualityMetrics(
                    total_chars=0, valid_chars=0, valid_ratio=0.0,
                    chinese_chars=0, english_chars=0, digit_chars=0,
                    space_chars=0, garbage_indicators=[f"error: {str(e)}"],
                    has_text_layer=False, page_count=0, sampled_pages=0
                ),
                reasons=[f"Quality check error: {str(e)}"]
            )
    
    def _analyze_text(
        self,
        text: str,
        has_text_layer: bool,
        page_count: int,
        sampled_pages: int
    ) -> QualityMetrics:
        """Analyze text and compute quality metrics.
        
        Args:
            text: Combined text from sampled pages
            has_text_layer: Whether text layer was found
            page_count: Total pages in document
            sampled_pages: Number of pages sampled
            
        Returns:
            QualityMetrics with analysis results
        """
        total_chars = len(text)
        
        # Character type counts
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        digit_chars = len(re.findall(r'\d', text))
        space_chars = len(re.findall(r'\s', text))
        
        # Valid characters: printable, not garbled
        # Remove control characters and common garbage patterns
        cleaned_text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
        # Remove box-drawing characters and other non-standard
        cleaned_text = re.sub(r'[\u2500-\u257f\u2580-\u259f]', '', cleaned_text)
        
        valid_chars = len(cleaned_text.replace('\n', '').replace('\r', '').replace('\t', ''))
        
        # Compute valid ratio
        non_space_chars = total_chars - space_chars
        valid_ratio = valid_chars / non_space_chars if non_space_chars > 0 else 0.0
        
        # Detect garbage indicators
        garbage_indicators = []
        
        # Check for excessive repeated characters (possible corruption)
        repeated_pattern = re.findall(r'(.)\1{10,}', text)
        if repeated_pattern:
            garbage_indicators.append("excessive_repetition")
        
        # Check for high ratio of special characters
        special_chars = len(re.findall(r'[^\w\s\u4e00-\u9fff]', text))
        if total_chars > 0 and special_chars / total_chars > 0.3:
            garbage_indicators.append("high_special_char_ratio")
        
        # Check for very low content
        if total_chars < self.min_chars:
            garbage_indicators.append("insufficient_content")
        
        # Check for no text layer
        if self.check_text_layer and not has_text_layer:
            garbage_indicators.append("no_text_layer")
        
        return QualityMetrics(
            total_chars=total_chars,
            valid_chars=valid_chars,
            valid_ratio=valid_ratio,
            chinese_chars=chinese_chars,
            english_chars=english_chars,
            digit_chars=digit_chars,
            space_chars=space_chars,
            garbage_indicators=garbage_indicators,
            has_text_layer=has_text_layer,
            page_count=page_count,
            sampled_pages=sampled_pages,
        )
    
    def _evaluate(self, metrics: QualityMetrics) -> QualityCheckResult:
        """Evaluate metrics and determine pass/fail.
        
        Args:
            metrics: QualityMetrics to evaluate
            
        Returns:
            QualityCheckResult with decision and reasons
        """
        reasons = []
        score = 1.0
        
        # Check valid ratio
        if metrics.valid_ratio < self.min_valid_ratio:
            reasons.append(
                f"Valid character ratio {metrics.valid_ratio:.1%} below threshold {self.min_valid_ratio:.1%}"
            )
            score *= metrics.valid_ratio / self.min_valid_ratio
        
        # Check text layer existence
        if self.check_text_layer and not metrics.has_text_layer:
            reasons.append("Document appears to be scan-only (no text layer)")
            score *= 0.3
        
        # Check garbage indicators
        if metrics.garbage_indicators:
            for indicator in metrics.garbage_indicators:
                reasons.append(f"Detected issue: {indicator}")
            score *= 0.5
        
        # Check insufficient content
        if metrics.total_chars < self.min_chars:
            reasons.append(
                f"Insufficient text content: {metrics.total_chars} chars (minimum: {self.min_chars})"
            )
            score *= 0.5
        
        passed = len(reasons) == 0 and score >= self.min_valid_ratio
        score = max(0.0, min(1.0, score))
        
        return QualityCheckResult(
            passed=passed,
            score=score,
            metrics=metrics,
            reasons=reasons
        )