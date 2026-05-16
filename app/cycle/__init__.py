"""
Trading Cycle Engine — isolated orchestration and execution.

This package contains the full trading cycle lifecycle:
  - orchestration/ — cycle control, state machine, lifecycle controller
  - phases/       — phase 1-6 implementations
  - trading_phase — trade execution and Kelly sizing
  - portfolio_gate — portfolio risk gating
  - core          — PipelineContext dataclass
  - attention_tracker — cycle attention tracking

Isolated from app.pipeline (data collection, analysis tools) so
the cycle logic can be tested and diagnosed independently.
"""
