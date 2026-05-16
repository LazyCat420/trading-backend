"""
Verification Module for Cognition V2.
Gates the pipeline on evidence sufficiency and evaluates hypotheses.
"""

from .sufficiency_gate import check_data_sufficiency

__all__ = ["check_data_sufficiency"]
