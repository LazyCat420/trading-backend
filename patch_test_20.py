import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target1 = """        with patch("app.cycle.phases.phase5_trading.alpaca.get_positions", return_value=[]):
            trade_result = await run_phase5_trading("""

replacement1 = """        with patch("app.cycle.trading_phase.get_portfolio", return_value={"cash": 100000, "position_count": 0, "positions": []}), \\
             patch("app.cycle.trading_phase.run_portfolio_allocator", new_callable=AsyncMock) as mock_allocator, \\
             patch("app.cycle.phases.phase5_trading.take_snapshot"):
            mock_allocator.return_value = {}
            trade_result = await run_phase5_trading("""

content = content.replace(target1, replacement1)

target2 = """        with patch("app.cycle.phases.phase5_trading.alpaca.get_positions", return_value=[]), \\
             patch("app.cycle.phases.phase5_trading.check_skip_rules", return_value=(True, "wash_sale", "Wash sale detected")):"""

replacement2 = """        with patch("app.cycle.trading_phase.get_portfolio", return_value={"cash": 100000, "position_count": 0, "positions": []}), \\
             patch("app.cycle.trading_phase.check_portfolio_gate", return_value={"blocked": True, "reason": "wash_sale", "warnings": []}), \\
             patch("app.cycle.phases.phase5_trading.take_snapshot"), \\
             patch("app.cycle.trading_phase.run_portfolio_allocator", new_callable=AsyncMock) as mock_allocator:
            mock_allocator.return_value = {}"""

content = content.replace(target2, replacement2)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)
