"""
Report Service — Generates markdown summaries for trading cycles.
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportService:
    @staticmethod
    def generate_report(summary: dict, results: list[dict] = None) -> str:
        """
        Generate a markdown report from a cycle summary dictionary.
        Returns the path to the generated report file.
        """
        cycle_id = summary.get("cycle_id", "unknown")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        # overwrite single file to prevent hundreds of files
        report_path = os.path.join(report_dir, "latest_cycle_report.md")

        status_emoji = "✅" if summary.get("status") == "done" else "❌"
        if summary.get("status") == "stopped":
            status_emoji = "⏹️"

        lines = [
            f"# Trading Cycle Report: {cycle_id}",
            f"**Generated At:** {timestamp}",
            f"**Status:** {status_emoji} {summary.get('status', 'unknown').upper()}",
            "",
            "## 📊 Metadata",
            f"- **Trigger:** {summary.get('trigger_type', 'manual')}",
            f"- **Duration:** {summary.get('elapsed_ms', 0)}ms",
            f"- **Jetson Healthy at Start:** {'Yes' if summary.get('jetson_healthy_start') else 'No'}",
            "",
            "## 📡 Data Collection",
            f"- **Tickers Requested:** {len(summary.get('tickers_requested', []))}",
            f"- **Tickers Final:** {len(summary.get('tickers_final', []))}",
            f"- **Collectors OK:** {summary.get('collector_ok', 0)}",
            f"- **Collectors Skipped:** {summary.get('collector_skipped', 0)}",
            f"- **Collectors Error:** {summary.get('collector_error', 0)}",
        ]

        if summary.get("collector_failures"):
            lines.append(f"- **Failures:** {', '.join(summary['collector_failures'])}")

        lines.extend(
            [
                "",
                "## 🧠 Analysis Findings",
                f"- **Tickers Analyzed:** {summary.get('analysis_results_count', 0)}",
                f"- **BUY:** {summary.get('buy_count', 0)}",
                f"- **SELL:** {summary.get('sell_count', 0)}",
                f"- **HOLD:** {summary.get('hold_count', 0)}",
                f"- **REVIEW:** {summary.get('review_count', 0)}",
            ]
        )

        # ADD DETAILED RESULTS BREAKDOWN
        if results:
            lines.extend(["", "## 🔍 Detailed Decisions"])
            for r in results:
                ticker = r.get("ticker", "UNKNOWN")
                action = r.get("action", "HOLD")
                confidence = r.get("confidence", 0)
                config_used = r.get("config_used", "N/A")

                emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
                lines.append(
                    f"\n### {emoji} {ticker} - {action} @ {confidence}% (Config: {config_used})"
                )

                # Execution / Sizing details
                exec_info = r.get("trade_executed") or r.get("trade_skipped")
                if exec_info:
                    status_type = "Executed" if r.get("trade_executed") else "Skipped"
                    # Include size or skip reason
                    qty = exec_info.get("qty")
                    price = exec_info.get("price")
                    reason = exec_info.get("reason", "")
                    if qty and price:
                        try:
                            lines.append(
                                f"- **Trade ({status_type}):** {float(qty):.2f} shares @ ${float(price):.2f}"
                            )
                        except (ValueError, TypeError):
                            lines.append(
                                f"- **Trade ({status_type}):** {qty} shares @ ${price}"
                            )
                    else:
                        lines.append(f"- **Trade ({status_type}):** {reason}")

                # Agent logic output
                agent_results = r.get("agent_results", {})
                if agent_results:
                    lines.append("\n**Agent Signals:**")
                    for agent_name, ares in agent_results.items():
                        resp = ares.get("response", "").strip().replace("\n", " ")
                        # limit response length to prevent gigantic markdown dumps
                        if len(resp) > 300:
                            resp = resp[:300] + "..."
                        lines.append(f"- *{agent_name.capitalize()}:* {resp}")

                # Model Rationale
                rationale = r.get("rationale") or "No rationale provided."
                lines.append("\n**Rationale:**")
                lines.append(f"> {rationale}")
                lines.append("\n---")

        # FETCH EXECUTION ERRORS
        try:
            from app.db.connection import get_db

            with get_db() as db:
                exec_errors = db.execute(
                    "SELECT phase, ticker, error_type, error_message FROM execution_errors WHERE cycle_id = %s",
                    [cycle_id],
                ).fetchall()

            if exec_errors:
                lines.extend(
                    [
                        "",
                        "## 🛑 Execution Errors",
                        "The following pipeline failures were caught during this cycle:",
                    ]
                )
                for err in exec_errors:
                    phase, err_ticker, err_type, err_msg = err
                    lines.append(
                        f"- **[{phase}] {err_ticker}**: {err_type} - {err_msg}"
                    )
        except Exception as e:
            logger.debug("[REPORT] Failed to fetch execution errors: %s", e)

        lines.extend(
            [
                "",
                "## 💸 Trade Execution Summary",
                f"- **Attempted:** {summary.get('trade_attempted', 0)}",
                f"- **Executed:** {summary.get('trade_executed', 0)}",
                f"- **Failed:** {summary.get('trade_failed', 0)}",
            ]
        )

        if summary.get("no_trade_reason"):
            lines.append(f"**Reason for No Trades:** `{summary['no_trade_reason']}`")

        if summary.get("primary_failure_reason"):
            lines.extend(
                [
                    "",
                    "## ⚠️ Failure Diagnosis",
                    "**Primary Reason:**",
                    f"> {summary['primary_failure_reason']}",
                ]
            )

        content = "\n".join(lines)

        # Sanitize surrogate characters that can leak in from LLM output
        # (surrogates are invalid in UTF-8 and crash the file write)
        content = content.encode("utf-8", errors="replace").decode("utf-8")

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("Generated cycle report: %s", report_path)
            return report_path
        except Exception as e:
            logger.error("Failed to generate report file: %s", e)
            return ""


report_service = ReportService()
