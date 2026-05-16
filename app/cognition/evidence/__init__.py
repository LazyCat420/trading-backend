"""
Evidence Fusion module for Cognition V2.
Extracts, normalizes, and fuses evidence into verifiable packets.
"""

from .packet_builder import build_evidence_packet
from .claim_extractor import extract_claims

__all__ = ["build_evidence_packet", "extract_claims"]
