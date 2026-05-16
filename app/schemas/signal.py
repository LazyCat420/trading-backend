from pydantic import BaseModel, Field

class AlgorithmSignal(BaseModel):
    """
    Universal interface for signals coming from various algorithm modules 
    (Markov chains, Bayesian scoring, etc.) before feeding into agents.
    """
    name: str = Field(..., description="The name of the algorithm generating this signal")
    value: float = Field(..., ge=-1.0, le=1.0, description="Signal value from -1 (Strong Sell) to +1 (Strong Buy)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level of the signal from 0 to 1")
    reasoning: str = Field(..., description="A short textual reasoning for the signal")
