# PII detection & masking
import re
from typing import Dict, Optional, Tuple


# Regex patterns for PII detection
PII_PATTERNS = {
    "phone": r"\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
}


def mask_pii(text: str, existing_mapping: Optional[Dict[str, str]] = None) -> Tuple[str, Dict[str, str]]:
    """
    Mask PII in text and return masked text with mapping.
    
    Args:
        text: The text to mask
        existing_mapping: Optional existing mapping to continue numbering from
    
    Returns:
        Tuple of (masked_text, mapping) where mapping is {token: original_value}
    """
    mapping = existing_mapping.copy() if existing_mapping else {}
    masked_text = text
    
    for pii_type, pattern in PII_PATTERNS.items():
        # Count existing tokens of this type to continue numbering
        existing_count = sum(1 for k in mapping if k.startswith(f"[{pii_type.upper()}_"))
        
        matches = re.findall(pattern, masked_text)
        for i, match in enumerate(matches):
            token = f"[{pii_type.upper()}_{existing_count + i + 1}]"
            mapping[token] = match
            masked_text = masked_text.replace(match, token, 1)
    
    return masked_text, mapping


def unmask_pii(text: str, mapping: Dict[str, str]) -> str:
    """
    Unmask PII in text using the provided mapping.
    
    Returns:
        Text with tokens replaced by original values.
    """
    unmasked_text = text
    for token, original in mapping.items():
        unmasked_text = unmasked_text.replace(token, original)
    
    return unmasked_text
