import pytest
from app.processors.ticker_extractor import (
    extract_tickers,
    get_ticker_symbols,
    _registry,
)

@pytest.fixture(autouse=True)
def setup_registry():
    # Ensure the registry is loaded before tests
    _registry.load()
    yield

@pytest.mark.parametrize("text, expected_symbols, excluded_symbols", [
    ("Ford Motor Company today announced new EV plans.", ["F"], ["FOR"]),
    ("Citi raises price target on NVDA.", ["C", "NVDA"], ["CITI"]),
    ("What are you waiting FOR?", [], ["F", "FOR"]),
    ("Are YOU READY for AI and the NEW world order?", [], ["YOU", "READY", "AI", "NEW"]),
    ("Bought $AAPL, MSFT, and NVDA!", ["AAPL", "MSFT", "NVDA"], []),
])
def test_ticker_extraction_cases(text, expected_symbols, excluded_symbols):
    """Verify ticker extraction, aliases, and exclusion lists using parameterized cases."""
    matches = extract_tickers(text)
    # Most valid matches in these sentences should have high confidence
    found_symbols = {m.symbol for m in matches if m.confidence >= 0.80}
    
    for sym in expected_symbols:
        assert sym in found_symbols
    
    # get_ticker_symbols is the public API which filters out low confidence
    final_tickers = get_ticker_symbols(text)
    for sym in excluded_symbols:
        assert sym not in final_tickers

def test_context_boosting():
    """Verify that ambiguous symbols get boosted by financial context."""
    text_no_context = "This is a random sentence about AAPL."
    matches = extract_tickers(text_no_context)
    aapl_conf = next((m.confidence for m in matches if m.symbol == "AAPL"), 0.0)
    
    # Add financial keywords: "stock", "portfolio"
    text_context = "This is a random sentence about AAPL stock for my portfolio."
    matches_ctx = extract_tickers(text_context)
    aapl_ctx_conf = next((m.confidence for m in matches_ctx if m.symbol == "AAPL"), 0.0)
    
    # Context should increase confidence
    assert aapl_ctx_conf > aapl_conf

def test_direct_syntax():
    """Verify that direct financial syntax gives massive confidence boosts."""
    text = "Just bought NVDA and grabbing calls on AMZN."
    matches = extract_tickers(text)
    nvda = next((m for m in matches if m.symbol == "NVDA"), None)
    amzn = next((m for m in matches if m.symbol == "AMZN"), None)
    
    assert nvda is not None
    assert nvda.confidence >= 0.90
    
    assert amzn is not None
    assert amzn.confidence >= 0.90
