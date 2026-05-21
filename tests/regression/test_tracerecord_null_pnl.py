"""
Regression test: TraceRecord NULL pnl_pct validation spam
=========================================================
Bug: The EvalWorker sweeps agent_traces and enriches them with data from
decision_outcomes. When pnl_pct or decision_confidence are NULL (the
normal state for unresolved decisions), the strict `float` type on the
TraceRecord Pydantic model raised ValidationError, producing 50+ warnings
every 2-minute sweep.

Fix: Made both fields Optional[float] with a @field_validator that coerces
None → 0.0, and added defensive `or 0.0` coalescing in eval_worker.py.

This test ensures the model never regresses back to rejecting None values.
"""

from app.autoresearch.eval_engine import TraceRecord, evaluate_trace, classify_failure


# ── Helpers ──────────────────────────────────────────────────────────

def _minimal_trace(**overrides) -> dict:
    """Return the smallest valid TraceRecord dict, with overrides applied."""
    base = {
        "id": "test-id-001",
        "run_id": "test-run-001",
    }
    base.update(overrides)
    return base


# ── Tests ────────────────────────────────────────────────────────────

class TestTraceRecordNullCoercion:
    """TraceRecord must accept None for nullable DB columns."""

    def test_pnl_pct_none_coerces_to_zero(self):
        record = TraceRecord(**_minimal_trace(pnl_pct=None))
        assert record.pnl_pct == 0.0

    def test_decision_confidence_none_coerces_to_zero(self):
        record = TraceRecord(**_minimal_trace(decision_confidence=None))
        assert record.decision_confidence == 0.0

    def test_both_none_simultaneously(self):
        record = TraceRecord(**_minimal_trace(
            pnl_pct=None,
            decision_confidence=None,
        ))
        assert record.pnl_pct == 0.0
        assert record.decision_confidence == 0.0

    def test_explicit_float_values_still_work(self):
        record = TraceRecord(**_minimal_trace(
            pnl_pct=3.14,
            decision_confidence=72.5,
        ))
        assert record.pnl_pct == 3.14
        assert record.decision_confidence == 72.5

    def test_defaults_when_omitted(self):
        record = TraceRecord(**_minimal_trace())
        assert record.pnl_pct == 0.0
        assert record.decision_confidence == 0.0


class TestEvalEngineWithNullFields:
    """Evaluate and classify should work when fields were coerced from None."""

    def test_evaluate_trace_with_coerced_nulls(self):
        record = TraceRecord(**_minimal_trace(
            pnl_pct=None,
            decision_confidence=None,
            stop_reason="completed",
        ))
        score = evaluate_trace(record)
        assert score["final_score"] >= 0.0
        assert score["completion_score"] == 40.0

    def test_classify_failure_with_coerced_nulls(self):
        record = TraceRecord(**_minimal_trace(
            pnl_pct=None,
            decision_confidence=None,
            decision_action="HOLD",
            stop_reason="budget_exhausted",
        ))
        score = evaluate_trace(record)
        bucket = classify_failure(record, score)
        # pnl_pct=0.0, so abs(0.0) > 2.0 is False → should NOT be hold_bias
        assert bucket != "hold_bias"

    def test_hold_bias_only_with_real_pnl(self):
        """hold_bias should only trigger when pnl_pct is genuinely > 2%."""
        record = TraceRecord(**_minimal_trace(
            pnl_pct=5.0,
            decision_confidence=80.0,
            decision_action="HOLD",
            stop_reason="completed",
        ))
        score = evaluate_trace(record)
        # Score is 100 (completed), so classify_failure returns None (>= 70)
        # Force a low score to test the bucket logic
        low_score = {**score, "final_score": 50.0}
        bucket = classify_failure(record, low_score)
        assert bucket == "hold_bias"
