"""
General Audit Tests — Environmental & Infrastructure.

Tests from the /general-audit checklist to verify system resilience
under adverse infrastructure conditions.
"""

import asyncio
import time
import logging
from contextlib import contextmanager
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# 1. Slow DB Test
#    Simulate the primary database responding in 5–10 seconds and verify
#    the processing cycle does not hang or crash.
# ────────────────────────────────────────────────────────────────────────


class TestSlowDB:
    """Verify cycle phases handle slow database responses without crash."""

    @staticmethod
    def _make_slow_cursor(delay_seconds: float = 6.0):
        """Create a mock cursor whose execute() sleeps to simulate slow DB."""
        cursor = MagicMock()

        def slow_execute(sql, params=None):
            time.sleep(delay_seconds)
            cursor.description = None
            return cursor

        cursor.execute.side_effect = slow_execute
        cursor.fetchone.return_value = (0,)
        cursor.fetchall.return_value = []
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        return cursor

    @staticmethod
    def _make_slow_get_db(delay_seconds: float = 6.0):
        """Create a patched get_db that yields a slow cursor."""
        slow_cursor = TestSlowDB._make_slow_cursor(delay_seconds)

        @contextmanager
        def slow_get_db():
            yield slow_cursor

        return slow_get_db, slow_cursor

    @pytest.mark.asyncio
    async def test_phase1_health_survives_slow_db(self):
        """Phase 1 health check should complete (not crash) with a slow DB.

        The DB takes ~1 second per query, but phase 1 should still run
        through reconciliation, triage, and directives without raising.
        """
        from app.cycle.context import CycleContext

        # Use a shorter delay for test speed — 1 second per query is enough
        # to prove the point without making the test take 30+ seconds.
        slow_get_db, slow_cursor = self._make_slow_get_db(delay_seconds=1.0)

        ctx = CycleContext(
            cycle_id="test-slow-db",
            tickers=["AAPL", "MSFT"],
            collect=False,
            analyze=False,
            trade=False,
        )
        cycle_summary = {}
        state = {"position_tickers": []}

        mock_health = AsyncMock(return_value={"jetson": True, "dgx_spark": True})

        with patch("app.cycle.phases.phase1_health.get_db", slow_get_db), \
             patch("app.cycle.phases.phase1_health.llm") as mock_llm, \
             patch("app.cycle.phases.phase1_health.check_stop_losses", new_callable=AsyncMock, return_value=[]), \
             patch("app.cycle.phases.phase1_health.check_take_profits", new_callable=AsyncMock, return_value=[]), \
             patch("app.cycle.phases.phase1_health.classify_tickers") as mock_triage, \
             patch("app.cycle.phases.phase1_health.increment_days_since_deep"), \
             patch("app.cycle.phases.phase1_health.flag_neglected_tickers", return_value=[]), \
             patch("app.cycle.phases.phase1_health.get_attention_summary", return_value={}):

            mock_llm.health_all = mock_health
            mock_triage.return_value = MagicMock(
                glance=["AAPL"], standard=["MSFT"], deep=[],
                summary=MagicMock(return_value="1 glance, 1 standard, 0 deep"),
            )

            from app.cycle.phases.phase1_health import run_phase1_health

            start = time.monotonic()
            # Should NOT raise — slow DB is tolerated, just slow
            await run_phase1_health(
                ctx=ctx,
                bot_id="test-bot",
                emit=lambda *a, **kw: None,
                cycle_summary=cycle_summary,
                state=state,
            )
            elapsed = time.monotonic() - start

            # Verify it completed (didn't crash)
            assert "jetson_healthy_start" in cycle_summary
            # It should have taken at least 1 second due to the slow DB
            assert elapsed >= 1.0, f"Expected slow path, but completed in {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_portfolio_gate_survives_slow_db(self):
        """Portfolio gate check_portfolio_gate should not crash if DB is slow."""
        slow_get_db, slow_cursor = self._make_slow_get_db(delay_seconds=1.0)

        # Portfolio gate reads positions and sector metadata
        slow_cursor.fetchone.return_value = None  # no sector found
        slow_cursor.fetchall.return_value = []    # no positions

        # Must patch at app.db.connection level since portfolio_gate AND
        # paper_trader both import get_db from app.db.connection.
        # Also mock paper_trader's portfolio helpers that do their own DB calls.
        with patch("app.db.connection.get_db", slow_get_db), \
             patch("app.cycle.portfolio_gate.get_db", slow_get_db), \
             patch("app.trading.paper_trader.get_portfolio_value", return_value=10000.0), \
             patch("app.trading.paper_trader._get_current_price", return_value=(150.0, 0.5)):
            from app.cycle.portfolio_gate import check_portfolio_gate

            start = time.monotonic()
            result = check_portfolio_gate(
                ticker="AAPL",
                action="BUY",
                bot_id="test-bot",
                confidence=80,
            )
            elapsed = time.monotonic() - start

            # Should complete without crash — even with slow DB
            assert isinstance(result, dict)
            assert "blocked" in result
            # It should have been slow due to DB delay
            assert elapsed >= 1.0, f"Expected slow path, but completed in {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_run_with_timeout_protects_against_slow_operations(self):
        """run_with_timeout should return fallback when operation exceeds timeout."""
        from app.utils.async_utils import run_with_timeout

        async def slow_operation():
            await asyncio.sleep(10)
            return "should never reach this"

        result = await run_with_timeout(
            slow_operation(),
            timeout=1.0,
            label="test-slow-op",
            fallback={"fallback": True},
        )

        assert result == {"fallback": True}

    @pytest.mark.asyncio
    async def test_state_manager_get_state_survives_slow_db(self):
        """Verify that PipelineStateDB.get_state doesn't hang when DB is slow.

        The state manager imports `from app.db import connection` and calls
        `connection.get_db()`. We patch at the module level.
        """
        slow_get_db, slow_cursor = self._make_slow_get_db(delay_seconds=1.0)
        slow_cursor.fetchone.return_value = None  # No saved state

        # State manager uses `connection.get_db()` so patch at the source module
        with patch("app.db.connection.get_db", slow_get_db):
            from app.cycle.orchestration.state_manager import PipelineStateDB

            start = time.monotonic()
            try:
                state = PipelineStateDB.get_state()
            except Exception:
                # Even if it fails, it shouldn't hang
                state = None
            elapsed = time.monotonic() - start

            # Should complete within reasonable time (slow but not hung)
            assert elapsed < 30.0, f"get_state took {elapsed:.2f}s — possible hang"


# ────────────────────────────────────────────────────────────────────────
# 2. Message Queue Silent Drop Test
#    Drop a pub/sub or queue message mid-cycle and verify the system
#    detects it and recovers.
#
#    In trading-service, the "message queue" is the asyncio.Queue that
#    feeds tickers from collection → analysis workers. Sentinels (None)
#    signal worker shutdown.
# ────────────────────────────────────────────────────────────────────────


class TestMessageQueueSilentDrop:
    """Verify the analysis queue handles dropped/missing messages gracefully."""

    @pytest.mark.asyncio
    async def test_priority_queue_sentinel_before_all_tickers_processed(self):
        """If a sentinel arrives before all tickers are consumed, workers
        should shut down cleanly without hanging."""
        from app.cycle.orchestration.priority_queue import PriorityAnalysisQueue

        q = PriorityAnalysisQueue()

        # Put 3 tickers then a sentinel
        await q.put("AAPL")
        await q.put("MSFT")
        await q.put("GOOG")
        await q.put(None)  # sentinel

        # Consume only 1 ticker, then hit sentinel
        t1 = await q.get()
        assert t1 is not None  # Should get a real ticker

        # The sentinel is still in the queue — remaining tickers are "dropped"
        # Workers should be able to drain the queue without hanging
        remaining = []
        while not q.empty():
            item = await q.get()
            remaining.append(item)

        # Verify we got the remaining tickers + sentinel
        assert None in remaining, "Sentinel should be in remaining items"
        # Total items consumed should be 4 (3 tickers + 1 sentinel)
        assert len(remaining) + 1 == 4  # +1 for the first one we consumed

    @pytest.mark.asyncio
    async def test_analysis_queue_worker_handles_early_sentinel(self):
        """Simulate a worker that receives a sentinel before processing
        all expected tickers. The worker should exit cleanly."""
        work_queue = asyncio.Queue()

        # Put sentinel immediately — simulates "dropped" ticker messages
        work_queue.put_nowait(None)

        processed = []

        async def mock_worker():
            while True:
                ticker = await work_queue.get()
                if ticker is None:
                    work_queue.task_done()
                    break
                processed.append(ticker)
                work_queue.task_done()

        # Worker should exit immediately without hanging
        await asyncio.wait_for(mock_worker(), timeout=2.0)
        assert processed == [], "Worker should process no tickers before sentinel"

    @pytest.mark.asyncio
    async def test_analysis_queue_partial_drain_then_sentinel(self):
        """Workers process some tickers, then sentinel arrives. Any
        unprocessed tickers between the last processed and sentinel
        remain in the queue but workers exit cleanly."""
        work_queue = asyncio.Queue()

        # Simulate: collection puts 5 tickers, but only 2 make it before sentinel
        work_queue.put_nowait("AAPL")
        work_queue.put_nowait("MSFT")
        work_queue.put_nowait(None)  # sentinel arrives early
        work_queue.put_nowait("GOOG")  # these arrive after sentinel
        work_queue.put_nowait("TSLA")

        processed = []
        exited_cleanly = False

        async def mock_worker():
            nonlocal exited_cleanly
            while True:
                ticker = await work_queue.get()
                if ticker is None:
                    work_queue.task_done()
                    exited_cleanly = True
                    break
                processed.append(ticker)
                work_queue.task_done()

        await asyncio.wait_for(mock_worker(), timeout=2.0)

        assert exited_cleanly, "Worker should exit cleanly on sentinel"
        assert processed == ["AAPL", "MSFT"], "Worker should process only pre-sentinel tickers"
        # Remaining items are still in the queue — not lost, just unprocessed
        assert work_queue.qsize() == 2, f"Expected 2 remaining, got {work_queue.qsize()}"

    @pytest.mark.asyncio
    async def test_multiple_workers_single_sentinel_propagation(self):
        """With multiple workers, a single sentinel should propagate to
        all workers (the analysis phase re-puts sentinel for next worker).

        This tests the actual pattern from phase4_analysis.py lines 158-162:
            if active_workers > 0:
                work_queue.put_nowait(None)
        """
        work_queue = asyncio.Queue()

        # 3 tickers + 1 sentinel for 2 workers
        work_queue.put_nowait("AAPL")
        work_queue.put_nowait("MSFT")
        work_queue.put_nowait("GOOG")
        work_queue.put_nowait(None)

        processed = []
        workers_exited = 0
        total_workers = 2
        active_count = total_workers
        lock = asyncio.Lock()

        async def mock_worker(worker_id):
            nonlocal workers_exited, active_count
            while True:
                ticker = await work_queue.get()
                if ticker is None:
                    async with lock:
                        active_count -= 1
                        workers_exited += 1
                        # Re-propagate sentinel for remaining workers
                        if active_count > 0:
                            work_queue.put_nowait(None)
                    work_queue.task_done()
                    break
                async with lock:
                    processed.append(ticker)
                work_queue.task_done()

        tasks = [asyncio.create_task(mock_worker(i)) for i in range(total_workers)]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

        assert workers_exited == total_workers, f"All {total_workers} workers should exit, got {workers_exited}"
        assert set(processed) == {"AAPL", "MSFT", "GOOG"}, f"All tickers should be processed: {processed}"

    @pytest.mark.asyncio
    async def test_queue_get_timeout_prevents_infinite_hang(self):
        """If no sentinel ever arrives (message dropped), a worker using
        asyncio.wait_for should not hang forever."""
        work_queue = asyncio.Queue()

        # Put 1 ticker but NO sentinel — simulates dropped sentinel
        work_queue.put_nowait("AAPL")

        processed = []
        timed_out = False

        async def mock_worker_with_timeout():
            nonlocal timed_out
            while True:
                try:
                    ticker = await asyncio.wait_for(work_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    timed_out = True
                    break
                if ticker is None:
                    work_queue.task_done()
                    break
                processed.append(ticker)
                work_queue.task_done()

        await asyncio.wait_for(mock_worker_with_timeout(), timeout=5.0)

        assert processed == ["AAPL"]
        assert timed_out, "Worker should timeout when no sentinel arrives"


# ────────────────────────────────────────────────────────────────────────
# 3. Container Restart Mid-Cycle Test
#    Verify that if the container is killed/restarted mid-cycle, it
#    recovers cleanly from the last checkpoint.
# ────────────────────────────────────────────────────────────────────────


class TestContainerRestart:
    """Verify system recovery when the backend is killed mid-cycle."""

    @pytest.mark.asyncio
    async def test_zombie_cycle_reset_on_boot_detects_checkpoint(self):
        """If a running cycle crashed (zombie) and a checkpoint exists,
        reset_on_boot should set status to 'interrupted'."""
        from app.services.pipeline_service import PipelineService

        # Mock state to be zombie
        zombie_state = {
            "status": "analyzing",
            "cycle_id": "zombie-123",
            "tickers": ["AAPL", "MSFT"],
            "progress": "processing",
        }

        mock_checkpoint = {
            "cycle_id": "zombie-123",
            "status": "interrupted",
            "completed_phases": ["collecting"],
            "completed_tickers": {"analyzing": []},
            "cycle_config": {"tickers": ["AAPL", "MSFT"]},
            "checkpoint_ts": "2026-05-30T00:00:00Z",
            "original_started_at": "2026-05-30T00:00:00Z",
        }

        saved_states = []

        def save_state_spy(state):
            saved_states.append(state.copy())

        # Reset mixin state
        PipelineService._state = zombie_state.copy()

        with patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_state", return_value=zombie_state), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_checkpoint", return_value=mock_checkpoint), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.save_state", side_effect=save_state_spy), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.expire_old_checkpoints"):

            PipelineService.reset_on_boot()

            # The state status and phase should have been set to 'interrupted'
            assert PipelineService._state["status"] == "interrupted"
            assert PipelineService._state["phase"] == "interrupted"
            assert "interrupted" in PipelineService._state["progress"]

            # Verify save_state was called with the updated state
            assert len(saved_states) > 0
            assert saved_states[-1]["status"] == "interrupted"

    @pytest.mark.asyncio
    async def test_zombie_cycle_reset_on_boot_no_checkpoint_resets_to_idle(self):
        """If a running cycle crashed but NO checkpoint exists,
        reset_on_boot should reset status to 'idle' and clean up orphaned data."""
        from app.services.pipeline_service import PipelineService

        # Mock state to be zombie
        zombie_state = {
            "status": "analyzing",
            "cycle_id": "zombie-456",
            "tickers": ["AAPL"],
        }

        PipelineService._state = zombie_state.copy()

        # Track sql statements executed
        sql_statements = []
        mock_cursor = MagicMock()

        def execute_spy(sql, params=None):
            sql_statements.append((sql, params))
            return mock_cursor

        mock_cursor.execute.side_effect = execute_spy
        mock_cursor.fetchone.return_value = None
        mock_cursor.fetchall.return_value = []
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        @contextmanager
        def spy_get_db():
            yield mock_cursor

        with patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_state", return_value=zombie_state), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_checkpoint", return_value=None), \
             patch("app.cycle.orchestration.state_manager.connection.get_db", spy_get_db), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.save_state"), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.expire_old_checkpoints"):

            PipelineService.reset_on_boot()

            # State should be reset to default (which has status='idle')
            assert PipelineService._state["status"] == "idle"

            # Check that database cleanups were called for orphaned data of zombie-456
            deletes = [sql for sql, params in sql_statements if "DELETE FROM" in sql]
            assert len(deletes) >= 3
            assert any("pipeline_events" in sql for sql in deletes)
            assert any("analysis_results" in sql for sql in deletes)
            assert any("debate_history" in sql for sql in deletes)
            for sql, params in sql_statements:
                if "DELETE FROM" in sql:
                    assert params == ["zombie-456"]

    @pytest.mark.asyncio
    async def test_resume_interrupted_cycle_determines_correct_phase_and_resumes(self):
        """Verify that resuming an interrupted cycle correctly recovers context
        and restarts execution from the correct phase."""
        from app.services.pipeline_service import PipelineService

        # Mock state to be interrupted
        interrupted_state = {
            "status": "interrupted",
            "cycle_id": "interrupted-789",
            "tickers": ["AAPL", "MSFT", "GOOG"],
            "collect_flag": True,
            "analyze_flag": True,
        }

        PipelineService._state = interrupted_state.copy()

        mock_checkpoint = {
            "cycle_id": "interrupted-789",
            "status": "interrupted",
            "completed_phases": ["collecting"],  # collecting completed -> resume_from is analyzing
            "completed_tickers": {"analyzing": ["AAPL"]},
            "cycle_config": {
                "tickers": ["AAPL", "MSFT", "GOOG"],
                "collect_flag": True,
                "analyze_flag": True,
                "macro_memo": "macro memo test",
            },
            "checkpoint_ts": "2026-05-30T00:00:00Z",
            "original_started_at": "2026-05-30T00:00:00Z",
        }

        # Mock db calls for fetch_already_analyzed & fetch_existing_results
        mock_cursor = MagicMock()
        query_count = 0

        def execute_spy(sql, params=None):
            nonlocal query_count
            query_count += 1
            return mock_cursor

        def fetchall_spy():
            if query_count == 1:
                return [("AAPL",)]
            elif query_count == 2:
                return [("AAPL", '{"ticker": "AAPL", "recommendation": "BUY"}')]
            return []

        mock_cursor.execute.side_effect = execute_spy
        mock_cursor.fetchall.side_effect = fetchall_spy
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)

        @contextmanager
        def spy_get_db():
            yield mock_cursor

        # We'll intercept `_run_cycle` call
        run_cycle_called_with = None

        async def dummy_run_cycle(ctx):
            nonlocal run_cycle_called_with
            run_cycle_called_with = ctx

        async def dummy_heartbeat(cycle_id):
            pass

        with patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_state", return_value=interrupted_state), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.get_checkpoint", return_value=mock_checkpoint), \
             patch("app.cycle.orchestration.lifecycle_controller.get_db", spy_get_db), \
             patch("app.cycle.orchestration.state_manager.PipelineStateDB.save_state"), \
             patch("app.cycle.orchestration.lifecycle_controller.PipelineStateDB.get_checkpoint", return_value=mock_checkpoint), \
             patch("app.services.pipeline_service.PipelineService._run_cycle", dummy_run_cycle), \
             patch("app.cycle.orchestration.lifecycle_controller.LifecycleControllerMixin._checkpoint_heartbeat", dummy_heartbeat):

            # Call background resume directly
            await PipelineService._background_resume_cycle("interrupted-789")

            # Let the loop run to execute tasks
            await asyncio.sleep(0.01)

            # Verify context passed to run_cycle
            assert run_cycle_called_with is not None
            assert run_cycle_called_with.cycle_id == "interrupted-789"
            assert run_cycle_called_with.tickers == ["AAPL", "MSFT", "GOOG"]
            assert run_cycle_called_with.resume_from == "analyzing"
            assert run_cycle_called_with.already_analyzed == ["AAPL"]
            assert len(run_cycle_called_with.existing_results) == 1
            assert run_cycle_called_with.existing_results[0]["ticker"] == "AAPL"
            assert run_cycle_called_with.macro_memo == "macro memo test"


# ────────────────────────────────────────────────────────────────────────
# 4. Inference / AI Service Overload Test
#    Verify that when the LLM service returns truncated, partial, or malformed
#    JSON responses (due to context overflow, timeouts, or model overload),
#    the agents recover gracefully using robust fallbacks.
# ────────────────────────────────────────────────────────────────────────


class TestInferenceOverload:
    """Verify agent resilience under partial/truncated LLM JSON responses."""

    @pytest.mark.asyncio
    async def test_pre_trade_agent_truncated_response(self):
        """Verify pre_trade_agent handles truncated JSON response without crashing."""
        from app.agents.pre_trade_agent import run_pre_trade

        # Simulate a JSON response that was cut off mid-sentence
        truncated_response = '{"decision": "VETO", "rationale": "High volatility and'

        with patch("app.agents.pre_trade_agent.run_agent") as mock_agent:
            mock_agent.return_value = {"response": truncated_response, "tokens_used": 10}
            result = await run_pre_trade("AAPL", 90, "cycle_1", "bot_1")

            # Should fallback to APPROVE with Kelly sizing
            assert result["decision"] == "APPROVE"
            assert "Kelly fallback" in result["rationale"]

    @pytest.mark.asyncio
    async def test_trade_execution_agent_truncated_response(self):
        """Verify trade_execution_agent handles truncated JSON response without crashing."""
        from app.agents.trade_execution_agent import run_trade_execution

        truncated_response = '{"decision": "VETO", "rationale": "Insufficient liquidity due to'

        with patch("app.agents.trade_execution_agent.run_agent") as mock_agent:
            mock_agent.return_value = {"response": truncated_response, "tokens_used": 10}

            # Action: BUY -> fallback is APPROVE / Kelly
            result = await run_trade_execution("AAPL", "BUY", 90, "cycle_1", "bot_1")
            assert result["decision"] == "APPROVE"
            assert "Kelly fallback" in result["rationale"]

            # Action: SELL -> fallback is APPROVE / 100% trim
            result_sell = await run_trade_execution("AAPL", "SELL", 90, "cycle_1", "bot_1")
            assert result_sell["decision"] == "APPROVE"
            assert result_sell["sell_pct"] == 100  # 100% trim fallback


    @pytest.mark.asyncio
    async def test_portfolio_allocator_agent_truncated_response(self):
        """Verify portfolio_allocator_agent handles truncated JSON response without crashing."""
        from app.agents.portfolio_allocator_agent import run_portfolio_allocator

        truncated_response = '{"allocations": [{"ticker": "AAPL", "weight": 0.5}, {"ticker": "MSFT'

        with patch("app.agents.portfolio_allocator_agent.run_agent") as mock_agent:
            mock_agent.return_value = {"response": truncated_response, "tokens_used": 15}

            result = await run_portfolio_allocator(
                [{"ticker": "AAPL", "action": "BUY", "confidence": 90}], "cycle_1", "bot_1"
            )
            # Should degrade to empty allocations dict
            assert result == {}

    @pytest.mark.asyncio
    async def test_post_mortem_auditor_agent_truncated_response(self):
        """Verify post_mortem_auditor_agent handles truncated JSON response without crashing."""
        from app.agents.post_mortem_auditor_agent import run_post_mortem

        truncated_response = '{"audit_findings": {"ticker": "AAPL", "performance":'

        with patch("app.agents.post_mortem_auditor_agent.run_agent") as mock_agent:
            mock_agent.return_value = {"response": truncated_response, "tokens_used": 12}

            result = await run_post_mortem("AAPL", 100.0, 110.0, 10.0, "cycle_1", "bot_1")
            assert result == {}

    @pytest.mark.asyncio
    async def test_meta_agent_truncated_response(self):
        """Verify meta_agent handles truncated JSON response without crashing."""
        from app.agents.meta_agent import generate_prompt

        truncated_response = '{"improved_prompt": "You are a trading bot assistant'

        with patch("app.agents.meta_agent.run_agent") as mock_agent:
            mock_agent.return_value = {"response": truncated_response, "tokens_used": 20}

            result = await generate_prompt("wins", "losses", "insights", "existing", "cycle_1", "bot_1")
            assert result == {}

    @pytest.mark.asyncio
    async def test_context_compressor_truncated_response(self):
        """Verify context_compressor generate_capsule degrades gracefully with truncated JSON."""
        from app.agents.context_compressor import generate_capsule

        # A truncated JSON which contains some plain text patterns that the regex fallback can extract
        truncated_response = '{"signal": "BUY", "confidence": 85, "summary": "Strong momentum'

        # When json parsing fails, it falls back to regex heuristic.
        # Let's ensure it extracts "BUY" and confidence 0.85
        result = await generate_capsule(
            {"response": truncated_response, "tokens_used": 10}, "test_agent", "cycle_1", "AAPL"
        )
        assert result.signal == "BUY"
        assert result.confidence == 0.85
        assert "Strong momentum" in result.summary



