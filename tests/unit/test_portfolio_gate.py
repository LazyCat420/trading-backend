import pytest
from unittest.mock import patch, MagicMock
from app.cycle.portfolio_gate import check_portfolio_gate

@pytest.fixture
def mock_dependencies():
    with patch("app.cycle.portfolio_gate._get_open_positions") as mock_open, \
         patch("app.cycle.portfolio_gate._get_ticker_sector") as mock_sector, \
         patch("app.cycle.portfolio_gate._load_thresholds") as mock_thresh, \
         patch("app.trading.paper_trader.get_portfolio") as mock_pf, \
         patch("app.trading.paper_trader._get_current_price") as mock_price:
        
        mock_thresh.return_value = (30.0, 8, 0.70)  # sector_cap, pos_cap, corr_thresh
        mock_sector.return_value = "Technology"
        mock_price.return_value = (None, None)      # force fallback to entry_price
        
        yield {
            "open": mock_open,
            "sector": mock_sector,
            "thresh": mock_thresh,
            "pf": mock_pf,
            "price": mock_price
        }

def test_portfolio_gate_new_position_under_cap(mock_dependencies):
    # Not held, under position cap
    deps = mock_dependencies
    deps["open"].return_value = []
    deps["pf"].return_value = {"cash": 10000.0, "positions": []}
    
    result = check_portfolio_gate("AAPL", "BUY", "bot_1")
    assert result["blocked"] is False
    assert result["reason"] is None

def test_portfolio_gate_new_position_at_cap_bypassed(mock_dependencies):
    # Not held, at position cap (8 positions) - should NOT block on position cap now
    deps = mock_dependencies
    sectors = ["SectorA", "SectorB", "SectorC", "SectorD", "SectorE", "SectorF", "SectorG", "SectorH"]
    deps["open"].return_value = [{"ticker": f"T{i}", "sector": sectors[i], "qty": 10, "entry_price": 10} for i in range(8)]
    deps["pf"].return_value = {"cash": 2000.0, "positions": []}
    
    result = check_portfolio_gate("AAPL", "BUY", "bot_1")
    assert result["blocked"] is False
    assert result["reason"] is None

def test_portfolio_gate_addition_at_cap_bypass(mock_dependencies):
    # Already held, at position cap (8 positions) -> should bypass position cap check
    deps = mock_dependencies
    held_positions = [{"ticker": f"T{i}", "sector": "Other", "qty": 10, "entry_price": 10.0} for i in range(7)]
    held_positions.append({"ticker": "AAPL", "sector": "Technology", "qty": 10, "entry_price": 100.0})
    deps["open"].return_value = held_positions
    deps["pf"].return_value = {"cash": 1000.0, "positions": held_positions}
    
    # AAPL value is 10 * 100 = 1000. Total portfolio is 1000 cash + 1000 AAPL + 700 other = 2700.
    # Concentration is 1000/2700 = 37% (which exceeds 20%, so it should be blocked due to concentration, not position limit)
    result = check_portfolio_gate("AAPL", "BUY", "bot_1")
    assert result["blocked"] is True
    assert "Concentration limit exceeded" in result["reason"]

def test_portfolio_gate_addition_under_concentration_cap(mock_dependencies):
    # Already held, under 20% concentration
    deps = mock_dependencies
    held_positions = [
        {"ticker": "AAPL", "sector": "Technology", "qty": 1, "entry_price": 100.0},
        {"ticker": "MSFT", "sector": "Other", "qty": 9, "entry_price": 100.0},
        {"ticker": "T1", "sector": "Other", "qty": 10, "entry_price": 100.0},
        {"ticker": "T2", "sector": "Other", "qty": 10, "entry_price": 100.0},
        {"ticker": "T3", "sector": "Other", "qty": 10, "entry_price": 100.0}
    ]
    deps["open"].return_value = held_positions
    deps["pf"].return_value = {"cash": 8000.0, "positions": held_positions}
    
    # AAPL value = 1 * 100 = 100. Total portfolio value = 8000 + 100 + 900 + 3000 = 12000.
    # AAPL concentration = 100 / 12000 = 0.83% (< 20%)
    # Sector concentration: 1 technology position (AAPL) out of 5 total = 20% (< 30% cap)
    result = check_portfolio_gate("AAPL", "BUY", "bot_1")
    assert result["blocked"] is False
    assert result["reason"] is None
    assert any("Already holding AAPL" in w for w in result["warnings"])
    assert any("representing 0.8% of portfolio" in w for w in result["warnings"])

