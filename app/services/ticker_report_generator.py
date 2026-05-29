"""
Ticker Report Generator — Produces detailed per-ticker markdown audit reports.

Each report captures the FULL pipeline output for a single ticker in a single
cycle, organized into collapsible sections. This gives humans a complete
audit trail to trace exactly how the system arrived at each trading decision.

Reports are saved:
  1. To disk: reports/<cycle_id>/<TICKER>.md
  2. To DB:   ticker_reports table

Usage:
    from app.services.ticker_report_generator import report_generator
    md = report_generator.generate_ticker_report(ticker, result, ...)
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class TickerReportGenerator:
    """Generates comprehensive per-ticker markdown reports."""

    REPORT_DIR = Path("reports")

    # ── Public API ───────────────────────────────────────────────────

    def generate_ticker_report(
        self,
        ticker: str,
        result: dict,
        cycle_id: str,
        cycle_summary: dict | None = None,
    ) -> str:
        """Generate a full markdown report for a single ticker.

        Args:
            ticker: Stock ticker symbol
            result: Full result dict from the pipeline (V1 or V2)
            cycle_id: Cycle identifier
            cycle_summary: Optional cycle-level summary dict

        Returns:
            Markdown string
        """
        report_data = result.get("_report_data", {})
        v2_meta = result.get("v2_metadata", {})
        action = result.get("action", "HOLD")
        confidence = result.get("confidence", 0)
        rationale = result.get("rationale", "No rationale provided.")
        config_used = result.get("config_used", "unknown")
        total_time = result.get("total_time_s", 0)
        total_tokens = result.get("total_tokens", 0)
        timestamp = result.get("timestamp", datetime.now(timezone.utc).isoformat())

        emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"

        sections = []

        # ── 1. Header ──────────────────────────────────────────────
        sections.append(self._section_header(
            ticker, action, confidence, emoji, config_used,
            total_time, total_tokens, cycle_id, timestamp,
        ))

        # ── 2. Decision Summary (not collapsed) ────────────────────
        sections.append(self._section_decision_summary(
            action, confidence, rationale, emoji,
        ))

        # ── 3. Pipeline Failure Diagnosis (if applicable) ──────────
        failure_diag = report_data.get("failure_diagnosis")
        if failure_diag:
            sections.append(self._section_failure_diagnosis(failure_diag))

        # ── 4. Pipeline Stages ─────────────────────────────────────
        stages = v2_meta.get("stages_completed", report_data.get("stages", []))
        stage_timings = report_data.get("stage_timings", {})
        if stages:
            sections.append(self._section_pipeline_stages(stages, stage_timings))

        # ── 5. Agent Signals ───────────────────────────────────────
        agent_insights = report_data.get("agent_insights", {})
        agent_results = result.get("agent_results", {})
        if agent_insights or agent_results:
            sections.append(self._section_agent_signals(agent_insights, agent_results))

        # ── 6. Debate Transcript ───────────────────────────────────
        debate = v2_meta.get("debate", {})
        debate_result_raw = report_data.get("debate_result")
        if debate or debate_result_raw:
            sections.append(self._section_debate(debate, debate_result_raw))

        # ── 7. Thesis Details ──────────────────────────────────────
        thesis_data = report_data.get("thesis")
        if thesis_data or v2_meta.get("thesis_action"):
            sections.append(self._section_thesis(thesis_data, v2_meta))

        # ── 8. Hallucination Check ─────────────────────────────────
        hall_data = report_data.get("hallucination_result")
        if hall_data:
            sections.append(self._section_hallucination(hall_data))

        # ── 9. Data Sources ────────────────────────────────────────
        present = result.get("data_sources", report_data.get("present_sources", []))
        missing = result.get("missing_sources", report_data.get("missing_sources", []))
        sufficiency = report_data.get("sufficiency")
        freshness = report_data.get("freshness_summary")
        if present or missing or sufficiency or freshness:
            sections.append(self._section_data_sources(
                present, missing, sufficiency, freshness,
            ))

        # ── 10. Evidence Packet ────────────────────────────────────
        evidence_facts = report_data.get("structured_facts", [])
        if evidence_facts:
            sections.append(self._section_evidence_packet(evidence_facts))

        # ── 11. Trade Execution ────────────────────────────────────
        trade_exec = result.get("trade_executed")
        trade_skip = result.get("trade_skipped")
        estimate = result.get("estimate")
        if trade_exec or trade_skip or estimate:
            sections.append(self._section_trade_execution(
                trade_exec, trade_skip, estimate,
            ))

        # ── 12. Config C / D Results ───────────────────────────────
        c_result = result.get("c_result")
        d_result = result.get("d_result")
        if c_result or d_result:
            sections.append(self._section_config_results(c_result, d_result))

        # ── 13. Memory Context ─────────────────────────────────────
        memory_brief = report_data.get("memory_brief", "")
        if memory_brief and memory_brief != "No prior memory.":
            sections.append(self._section_memory(memory_brief))

        report = "\n\n".join(sections)
        return self._sanitize(report)

    def generate_cycle_summary_report(
        self,
        cycle_id: str,
        results: list[dict],
        cycle_summary: dict | None = None,
    ) -> str:
        """Generate a roll-up summary report for the entire cycle."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        summary = cycle_summary or {}

        buys = [r for r in results if r.get("action") == "BUY"]
        sells = [r for r in results if r.get("action") == "SELL"]
        holds = [r for r in results if r.get("action") == "HOLD"]
        errors = [r for r in results if r.get("error")]

        total_tokens = sum(r.get("total_tokens", 0) for r in results)
        total_time = sum(r.get("total_time_s", 0) for r in results)

        lines = [
            f"# 📊 Cycle Summary Report: {cycle_id}",
            f"**Generated:** {timestamp}",
            f"**Status:** {summary.get('status', 'completed').upper()}",
            f"**Tickers Analyzed:** {len(results)}",
            f"**Total Tokens:** {total_tokens:,}",
            f"**Total Pipeline Time:** {total_time:.1f}s",
            "",
            "---",
            "",
            "## Action Breakdown",
            f"| Action | Count | Tickers |",
            f"|--------|-------|---------|",
            f"| 🟢 BUY | {len(buys)} | {', '.join(r.get('ticker', '?') for r in buys) or '—'} |",
            f"| 🔴 SELL | {len(sells)} | {', '.join(r.get('ticker', '?') for r in sells) or '—'} |",
            f"| 🟡 HOLD | {len(holds)} | {', '.join(r.get('ticker', '?') for r in holds) or '—'} |",
            f"| ❌ ERROR | {len(errors)} | {', '.join(r.get('ticker', '?') for r in errors) or '—'} |",
        ]

        # Per-ticker summary table
        lines.extend([
            "",
            "## Per-Ticker Summary",
            "",
            "| Ticker | Action | Confidence | Config | Time (s) | Tokens |",
            "|--------|--------|------------|--------|----------|--------|",
        ])
        for r in sorted(results, key=lambda x: x.get("confidence", 0), reverse=True):
            t = r.get("ticker", "?")
            a = r.get("action", r.get("error", "ERROR"))
            c = r.get("confidence", 0)
            cfg = r.get("config_used", "?")
            tm = r.get("total_time_s", 0)
            tok = r.get("total_tokens", 0)
            emoji = "🟢" if a == "BUY" else "🔴" if a == "SELL" else "🟡" if a == "HOLD" else "❌"
            lines.append(f"| {emoji} {t} | {a} | {c}% | {cfg} | {tm:.1f} | {tok:,} |")

        # Trade execution summary
        if summary.get("trade_attempted") or summary.get("trade_executed"):
            lines.extend([
                "",
                "## Trade Execution",
                f"- **Attempted:** {summary.get('trade_attempted', 0)}",
                f"- **Executed:** {summary.get('trade_executed', 0)}",
                f"- **Failed:** {summary.get('trade_failed', 0)}",
            ])

        # Errors
        if errors:
            lines.extend([
                "",
                "## ❌ Pipeline Errors",
            ])
            for r in errors:
                lines.append(f"- **{r.get('ticker', '?')}**: {r.get('error', 'Unknown error')}")

        return self._sanitize("\n".join(lines))

    def save_reports(
        self,
        cycle_id: str,
        results: list[dict],
        cycle_summary: dict | None = None,
    ) -> dict:
        """Generate and save all reports for a cycle.

        Returns dict with counts: {"ticker_reports": N, "summary": bool, "errors": [...]}
        """
        report_count = 0
        errors = []
        cycle_dir = self.REPORT_DIR / cycle_id
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # Generate per-ticker reports
        for result in results:
            ticker = result.get("ticker", "")
            if not ticker:
                continue
            try:
                md = self.generate_ticker_report(
                    ticker=ticker,
                    result=result,
                    cycle_id=cycle_id,
                    cycle_summary=cycle_summary,
                )

                # Save to file
                file_path = cycle_dir / f"{ticker}.md"
                file_path.write_text(md, encoding="utf-8")

                # Save to DB
                self._save_to_db(
                    cycle_id=cycle_id,
                    ticker=ticker,
                    report_markdown=md,
                    result=result,
                    is_summary=False,
                )

                report_count += 1
            except Exception as e:
                logger.warning("[REPORT] Failed to generate report for %s: %s", ticker, e)
                errors.append(f"{ticker}: {e}")

        # Generate cycle summary
        summary_saved = False
        try:
            summary_md = self.generate_cycle_summary_report(
                cycle_id=cycle_id,
                results=results,
                cycle_summary=cycle_summary,
            )
            summary_path = cycle_dir / "summary.md"
            summary_path.write_text(summary_md, encoding="utf-8")

            self._save_to_db(
                cycle_id=cycle_id,
                ticker="__SUMMARY__",
                report_markdown=summary_md,
                result={"action": "SUMMARY", "confidence": 0},
                is_summary=True,
            )
            summary_saved = True
        except Exception as e:
            logger.warning("[REPORT] Failed to generate cycle summary: %s", e)
            errors.append(f"summary: {e}")

        logger.info(
            "[REPORT] Generated %d ticker reports + summary=%s for cycle %s",
            report_count, summary_saved, cycle_id,
        )
        return {
            "ticker_reports": report_count,
            "summary": summary_saved,
            "errors": errors,
        }

    # ── Private: Section Generators ──────────────────────────────────

    def _section_header(
        self, ticker, action, confidence, emoji, config_used,
        total_time, total_tokens, cycle_id, timestamp,
    ) -> str:
        return "\n".join([
            f"# {emoji} {ticker} — {action} @ {confidence}%",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| **Cycle ID** | `{cycle_id}` |",
            f"| **Timestamp** | {timestamp} |",
            f"| **Config** | {config_used} |",
            f"| **Total Time** | {total_time:.1f}s |",
            f"| **Total Tokens** | {total_tokens:,} |",
        ])

    def _section_decision_summary(self, action, confidence, rationale, emoji) -> str:
        return "\n".join([
            f"## {emoji} Decision: {action} @ {confidence}%",
            "",
            rationale,
        ])

    def _section_failure_diagnosis(self, diag: dict) -> str:
        lines = [
            "## ⚠️ Pipeline Failure Diagnosis",
            "",
            f"**Failure Type:** {diag.get('failure_type', 'UNKNOWN')}",
            f"**Stages Completed:** {', '.join(diag.get('stages_completed', [])) or 'None'}",
            f"**Stages Failed:** {', '.join(diag.get('stages_failed', [])) or 'None'}",
        ]
        if diag.get("meta_orchestrator_agents"):
            lines.append(f"**MetaOrchestrator Agents:** {diag['meta_orchestrator_agents']}")
        if diag.get("tokens_per_stage"):
            lines.append("")
            lines.append("**Tokens Per Stage:**")
            for stage, tokens in diag["tokens_per_stage"].items():
                lines.append(f"- {stage}: {tokens:,}")
        if diag.get("error_chain"):
            lines.append("")
            lines.append("**Error Chain:**")
            for err in diag["error_chain"]:
                lines.append(f"- {err}")
        if diag.get("data_available"):
            lines.append("")
            lines.append(f"**Data Available:** {diag['data_available']}")
        return "\n".join(lines)

    def _section_pipeline_stages(self, stages: list, timings: dict) -> str:
        lines = [
            "<details>",
            "<summary><strong>🔧 Pipeline Stages</strong></summary>",
            "",
            "| Stage | Elapsed (ms) |",
            "|-------|-------------|",
        ]
        for stage in stages:
            ms = timings.get(stage, "—")
            lines.append(f"| {stage} | {ms} |")
        lines.append("")
        lines.append("</details>")
        return "\n".join(lines)

    def _section_agent_signals(self, insights: dict, agent_results: dict) -> str:
        lines = [
            "<details>",
            "<summary><strong>🤖 Agent Signals</strong></summary>",
            "",
        ]
        # Full insights (V2 path — untruncated)
        if insights:
            for agent_name, insight in insights.items():
                lines.append(f"### {agent_name.upper()}")
                lines.append("")
                lines.append("```")
                text = str(insight) if not isinstance(insight, str) else insight
                lines.append(text[:5000])
                lines.append("```")
                lines.append("")

        # Agent results (V1 path or fallback)
        if agent_results and not insights:
            for agent_name, ares in agent_results.items():
                resp = ares.get("response", "") if isinstance(ares, dict) else str(ares)
                tokens = ares.get("tokens", 0) if isinstance(ares, dict) else 0
                lines.append(f"### {agent_name.upper()} ({tokens:,} tokens)")
                lines.append("")
                lines.append("```")
                lines.append(resp[:5000])
                lines.append("```")
                lines.append("")

        lines.append("</details>")
        return "\n".join(lines)

    def _section_debate(self, debate_meta: dict, debate_result_raw) -> str:
        lines = [
            "<details>",
            "<summary><strong>⚔️ Adversarial Debate</strong></summary>",
            "",
        ]

        # Judge verdict
        judge_action = debate_meta.get("judge_action", "N/A")
        judge_conf = debate_meta.get("judge_confidence", "N/A")
        winner = debate_meta.get("winning_side", "N/A")
        integrity = debate_meta.get("integrity_status", "N/A")
        key_factor = debate_meta.get("key_deciding_factor", "N/A")

        lines.extend([
            "### Judge Verdict",
            f"- **Action:** {judge_action} @ {judge_conf}%",
            f"- **Winner:** {winner}",
            f"- **Integrity:** {integrity}",
            f"- **Key Factor:** {key_factor}",
            "",
        ])

        # Claims verification
        bull_verified = debate_meta.get("bull_claims_verified", "?/?")
        bear_verified = debate_meta.get("bear_claims_verified", "?/?")
        unverified = debate_meta.get("unverified_claims", 0)

        lines.extend([
            "### Evidence Verification",
            f"- Bull claims verified: {bull_verified}",
            f"- Bear claims verified: {bear_verified}",
            f"- Unverified claims rejected: {unverified}",
            "",
        ])

        # Full transcript (if available)
        transcript = debate_meta.get("transcript", "")
        if transcript:
            lines.extend([
                "### Full Transcript",
                "",
                "```",
                str(transcript)[:10000],
                "```",
                "",
            ])

        # Persona outcomes
        personas = debate_meta.get("persona_outcomes", {})
        if personas:
            lines.extend(["### Persona Outcomes", ""])
            for name, outcome in personas.items():
                lines.append(f"- **{name}**: {outcome}")
            lines.append("")

        # Original thesis status (for held positions)
        ots = debate_meta.get("original_thesis_status", "")
        if ots and ots != "NOT_HELD":
            lines.extend([
                "### Original Thesis Status",
                f"- **Status:** {ots}",
                f"- **Explanation:** {debate_meta.get('original_thesis_explanation', 'N/A')}",
                "",
            ])

        lines.append("</details>")
        return "\n".join(lines)

    def _section_thesis(self, thesis_data, v2_meta: dict) -> str:
        lines = [
            "<details>",
            "<summary><strong>📝 Thesis Details</strong></summary>",
            "",
        ]

        if thesis_data:
            # Thesis object attributes
            t_action = getattr(thesis_data, "action", None) or v2_meta.get("thesis_action", "?")
            t_conf = getattr(thesis_data, "confidence", None) or v2_meta.get("thesis_confidence", "?")
            t_rationale = getattr(thesis_data, "rationale", "") or ""
            claims = getattr(thesis_data, "core_claims", []) or []
            weaknesses = getattr(thesis_data, "weaknesses", []) or v2_meta.get("thesis_weaknesses", [])

            lines.extend([
                f"**Action:** {t_action} @ {t_conf}%",
                "",
            ])

            if t_rationale:
                lines.extend([
                    "**Rationale:**",
                    t_rationale[:3000],
                    "",
                ])

            if claims:
                lines.append("**Core Claims:**")
                for i, claim in enumerate(claims, 1):
                    claim_text = str(claim) if not isinstance(claim, str) else claim
                    lines.append(f"{i}. {claim_text[:500]}")
                lines.append("")

            if weaknesses:
                lines.append("**Weaknesses:**")
                for w in weaknesses:
                    w_text = str(w) if not isinstance(w, str) else w
                    lines.append(f"- ⚠️ {w_text[:500]}")
                lines.append("")
        else:
            lines.extend([
                f"**Action:** {v2_meta.get('thesis_action', '?')} @ {v2_meta.get('thesis_confidence', '?')}%",
                "",
            ])
            weaknesses = v2_meta.get("thesis_weaknesses", [])
            if weaknesses:
                lines.append("**Weaknesses:**")
                for w in weaknesses:
                    lines.append(f"- ⚠️ {w}")
                lines.append("")

        lines.append("</details>")
        return "\n".join(lines)

    def _section_hallucination(self, hall_data: dict) -> str:
        rejected = hall_data.get("rejected", False)
        reason = hall_data.get("rejection_reason", "")
        hallucinations = hall_data.get("hallucinations", [])
        status_emoji = "❌ REJECTED" if rejected else "✅ PASSED"

        lines = [
            "<details>",
            "<summary><strong>🔍 Hallucination Check — " + status_emoji + "</strong></summary>",
            "",
            f"**Status:** {status_emoji}",
        ]

        if rejected:
            lines.append(f"**Rejection Reason:** {reason}")
            lines.append("")

        if hallucinations:
            lines.append(f"**Flagged Items ({len(hallucinations)}):**")
            for h in hallucinations:
                lines.append(f"- {h}")
            lines.append("")

        if not rejected and not hallucinations:
            lines.append("No hallucinations detected.")
            lines.append("")

        lines.append("</details>")
        return "\n".join(lines)

    def _section_data_sources(
        self, present: list, missing: list,
        sufficiency, freshness,
    ) -> str:
        lines = [
            "<details>",
            "<summary><strong>📡 Data Sources</strong></summary>",
            "",
        ]

        if sufficiency:
            status = getattr(sufficiency, "status", str(sufficiency)) if not isinstance(sufficiency, str) else sufficiency
            lines.append(f"**Sufficiency Status:** {status}")
            warnings = getattr(sufficiency, "warnings", []) if not isinstance(sufficiency, (str, dict)) else []
            if warnings:
                lines.append(f"**Warnings:** {'; '.join(str(w) for w in warnings)}")
            lines.append("")

        if present:
            lines.append(f"**Present ({len(present)}):** {', '.join(str(s) for s in present)}")
        if missing:
            lines.append(f"**Missing ({len(missing)}):** {', '.join(str(s) for s in missing)}")

        if freshness:
            oldest = getattr(freshness, "oldest_timestamp", None)
            newest = getattr(freshness, "newest_timestamp", None)
            if oldest and newest:
                lines.append("")
                lines.append(f"**Data Timeframe:** {oldest} → {newest}")

        lines.append("")
        lines.append("</details>")
        return "\n".join(lines)

    def _section_evidence_packet(self, facts: list) -> str:
        lines = [
            "<details>",
            "<summary><strong>📦 Evidence Packet (Top Facts)</strong></summary>",
            "",
            "| Field | Value | Source |",
            "|-------|-------|--------|",
        ]

        # Show top 50 facts to keep report manageable
        for fact in facts[:50]:
            if isinstance(fact, dict):
                field = fact.get("field_name", "?")
                value = str(fact.get("value", ""))[:100]
                source = fact.get("source", "?")
            elif hasattr(fact, "field_name"):
                field = fact.field_name
                value = str(fact.value)[:100] if fact.value is not None else ""
                source = fact.source if hasattr(fact, "source") else "?"
            else:
                field = str(fact)[:50]
                value = ""
                source = "?"
            # Escape pipes in values
            value = value.replace("|", "\\|")
            field = str(field).replace("|", "\\|")
            source = str(source).replace("|", "\\|")
            lines.append(f"| {field} | {value} | {source} |")

        if len(facts) > 50:
            lines.append(f"| ... | *{len(facts) - 50} more facts omitted* | |")

        lines.append("")
        lines.append("</details>")
        return "\n".join(lines)

    def _section_trade_execution(self, executed: dict, skipped: dict, estimate: dict) -> str:
        lines = [
            "<details>",
            "<summary><strong>💰 Trade Execution</strong></summary>",
            "",
        ]

        if executed:
            lines.append("### ✅ Trade Executed")
            lines.append(f"- **Action:** {executed.get('action', '?')}")
            lines.append(f"- **Ticker:** {executed.get('ticker', '?')}")
            qty = executed.get("qty")
            price = executed.get("price")
            if qty is not None and price is not None:
                try:
                    lines.append(f"- **Quantity:** {float(qty):.2f} shares")
                    lines.append(f"- **Price:** ${float(price):.2f}")
                    lines.append(f"- **Total:** ${float(qty) * float(price):.2f}")
                except (ValueError, TypeError):
                    lines.append(f"- **Quantity:** {qty}")
                    lines.append(f"- **Price:** {price}")
            pnl = executed.get("pnl_pct")
            if pnl is not None:
                lines.append(f"- **P&L:** {pnl}%")
            lines.append("")

        if skipped:
            lines.append("### ⏭️ Trade Skipped")
            lines.append(f"- **Reason:** {skipped.get('reason', 'Unknown')}")
            lines.append(f"- **Ticker:** {skipped.get('ticker', '?')}")
            lines.append("")

        if estimate:
            lines.append("### 📐 Trade Estimate")
            for k, v in estimate.items():
                lines.append(f"- **{k}:** {v}")
            lines.append("")

        lines.append("</details>")
        return "\n".join(lines)

    def _section_config_results(self, c_result: dict, d_result: dict) -> str:
        lines = [
            "<details>",
            "<summary><strong>⚙️ Config C/D Results</strong></summary>",
            "",
        ]

        if c_result:
            lines.extend([
                "### Config C (Initial Analysis)",
                f"- **Action:** {c_result.get('action', '?')}",
                f"- **Confidence:** {c_result.get('confidence', '?')}%",
            ])
            if c_result.get("rationale"):
                lines.append(f"- **Rationale:** {c_result['rationale'][:500]}")
            lines.append("")

        if d_result:
            lines.extend([
                "### Config D (Escalation / Debate)",
                f"- **Action:** {d_result.get('action', '?')}",
                f"- **Confidence:** {d_result.get('confidence', '?')}%",
            ])
            ots = d_result.get("original_thesis_status")
            if ots and ots != "NOT_HELD":
                lines.append(f"- **Original Thesis Status:** {ots}")
                lines.append(f"- **Explanation:** {d_result.get('original_thesis_explanation', 'N/A')}")
            lines.append("")

        lines.append("</details>")
        return "\n".join(lines)

    def _section_memory(self, memory_brief: str) -> str:
        return "\n".join([
            "<details>",
            "<summary><strong>🧠 Memory Context</strong></summary>",
            "",
            memory_brief[:2000],
            "",
            "</details>",
        ])

    # ── Private: Persistence ─────────────────────────────────────────

    def _save_to_db(
        self,
        cycle_id: str,
        ticker: str,
        report_markdown: str,
        result: dict,
        is_summary: bool = False,
    ) -> None:
        """Save a report to the ticker_reports DB table.

        Uses parameterized queries to prevent SQL injection.
        """
        try:
            from app.db.connection import get_db

            report_id = str(uuid.uuid4())
            action = result.get("action", "HOLD")
            confidence = result.get("confidence", 0)

            # Build a compact summary for the JSONB column
            result_summary = {
                "action": action,
                "confidence": confidence,
                "config_used": result.get("config_used", ""),
                "total_tokens": result.get("total_tokens", 0),
                "total_time_s": result.get("total_time_s", 0),
                "escalated": result.get("escalated", False),
            }

            with get_db() as db:
                db.execute(
                    """
                    INSERT INTO ticker_reports
                    (id, cycle_id, ticker, action, confidence,
                     report_markdown, result_summary, is_summary, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cycle_id, ticker) DO UPDATE SET
                        action = EXCLUDED.action,
                        confidence = EXCLUDED.confidence,
                        report_markdown = EXCLUDED.report_markdown,
                        result_summary = EXCLUDED.result_summary,
                        created_at = EXCLUDED.created_at
                    """,
                    [
                        report_id,
                        cycle_id,
                        ticker,
                        action,
                        confidence,
                        report_markdown,
                        json.dumps(result_summary),
                        is_summary,
                        datetime.now(timezone.utc).isoformat(),
                    ],
                )
        except Exception as e:
            logger.warning(
                "[REPORT] DB save failed for %s/%s: %s", cycle_id, ticker, e,
            )

    # ── Private: Utilities ───────────────────────────────────────────

    @staticmethod
    def _sanitize(content: str) -> str:
        """Sanitize surrogate characters that can leak from LLM output."""
        return content.encode("utf-8", errors="replace").decode("utf-8")


# Module-level singleton
report_generator = TickerReportGenerator()
