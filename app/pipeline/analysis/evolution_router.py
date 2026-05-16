"""
Evolution Router — Orchestrates the autonomous evolutionary fixing pipeline.

Parses AutoResearch JSON reports and spawns Debate Council threads
targeted at the specific files/prompts/scrapers that are failing.
Uses the Target Mapping Engine to resolve issue names → real files.
"""

import asyncio
import json
import logging

from app.db.connection import get_db
from app.cognition.evolution.debate import council

logger = logging.getLogger(__name__)


class EvolutionRouter:
    # Number of consecutive rejections before the circuit breaker trips
    MAX_CONSECUTIVE_REJECTIONS = 3
    # Graduated backoff: skip N cycles after each consecutive rejection tier
    # Tier 1 (3 rejections): skip 2 cycles, Tier 2 (6): skip 4, Tier 3 (9+): skip 8, then reset
    BACKOFF_TIERS = {3: 2, 6: 4, 9: 8}

    def _is_circuit_open(self, target_type: str, target_name: str) -> bool:
        """Check if a target should be skipped due to graduated backoff.

        Instead of permanently blocking a target after 3 rejections, uses
        exponential backoff: skip 2 → 4 → 8 cycles, then resets.
        Returns True if the circuit is OPEN (should skip debate).
        """
        try:
            with get_db() as db:
                # Count total consecutive rejections (look back further)
                rows = db.execute(
                    "SELECT status, created_at FROM pending_evolution_fixes "
                    "WHERE target_type = %s AND target_name = %s "
                    "ORDER BY created_at DESC LIMIT 12",
                    [target_type, target_name],
                ).fetchall()

                if not rows:
                    return False  # No history at all

                # Count consecutive rejections from most recent
                consecutive_rejections = 0
                for r in rows:
                    if r[0] == "rejected":
                        consecutive_rejections += 1
                    else:
                        break  # Chain broken by a non-rejection

                if consecutive_rejections < self.MAX_CONSECUTIVE_REJECTIONS:
                    return False  # Not enough rejections to trigger backoff

                # After 12+ consecutive rejections, reset the counter (fresh start)
                if consecutive_rejections >= 12:
                    logger.info(
                        "[EVO-ROUTER] Backoff reset for %s:%s after %d consecutive rejections",
                        target_type, target_name, consecutive_rejections,
                    )
                    return False

                # Find the appropriate backoff tier
                skip_cycles = 2  # Default
                for threshold, backoff in sorted(self.BACKOFF_TIERS.items()):
                    if consecutive_rejections >= threshold:
                        skip_cycles = backoff

                # Check how many cycles have passed since the last rejection
                last_rejection_at = rows[0][1]
                cycles_since = db.execute(
                    "SELECT COUNT(*) FROM autoresearch_reports "
                    "WHERE created_at > %s AND status = 'done'",
                    [last_rejection_at],
                ).fetchone()[0]

                if cycles_since < skip_cycles:
                    logger.debug(
                        "[EVO-ROUTER] Backoff active for %s:%s — %d/%d cycles remaining",
                        target_type, target_name, skip_cycles - cycles_since, skip_cycles,
                    )
                    return True  # Still in backoff period

                return False  # Backoff expired, allow retry
        except Exception:
            return False  # Fail-open: if we can't check, allow the debate

    async def run_router(self, cycle_id: str):
        """Entry point. Called by autoresearch after the reflection phase."""
        logger.info("[EVO-ROUTER] Routing evolution fixes for cycle %s", cycle_id)

        with get_db() as db:
            try:
                row = db.execute(
                    "SELECT data_gaps, decision_issues, llm_issues "
                    "FROM autoresearch_reports WHERE cycle_id = %s",
                    [cycle_id],
                ).fetchone()

                if not row:
                    logger.warning(
                        "[EVO-ROUTER] No AutoResearch report for %s", cycle_id
                    )
                    return

                data_gaps = json.loads(row[0] or "[]")
                decision_issues = json.loads(row[1] or "[]")
                llm_issues = json.loads(row[2] or "[]")

                tasks: list = []
                skipped_circuit: list[str] = []

                # ── 1. Prompt Evolution (LLM Issues) ──
                # Deduplicate by agent name so we don't spawn 5 debates for the same prompt
                seen_agents: set[str] = set()
                for issue in llm_issues:
                    agent_name = issue.get("agent", "unknown")
                    if agent_name in seen_agents:
                        continue
                    seen_agents.add(agent_name)

                    # Circuit breaker: skip if recently rejected too many times
                    if self._is_circuit_open("prompt", agent_name):
                        skipped_circuit.append(f"prompt:{agent_name}")
                        continue

                    issue_desc = (
                        f"Agent '{agent_name}' is failing.\n"
                        f"Issue: {issue.get('issue', 'unknown')}\n"
                        f"Occurrence count: {issue.get('count', '%s')}\n"
                        f"Model: {issue.get('model', 'unknown')}"
                    )

                    tasks.append(
                        council.run_debate(
                            cycle_id=cycle_id,
                            target_type="prompt",
                            target_name=agent_name,
                            issue_description=issue_desc,
                            # current_content=None → Mapping Engine will resolve it
                        )
                    )

                # ── 2. Scraper Evolution (Data Gaps) ──
                seen_sources: set[str] = set()
                for gap in data_gaps:
                    ticker = gap.get("ticker", "%s%s%s")
                    sources = gap.get("missing_sources", [])
                    recommendation = gap.get("recommendation", "")

                    for source in sources:
                        if source in seen_sources:
                            continue
                        seen_sources.add(source)

                        # Circuit breaker: skip if recently rejected too many times
                        if self._is_circuit_open("scraper", source):
                            skipped_circuit.append(f"scraper:{source}")
                            continue

                        issue_desc = (
                            f"Data source '{source}' is failing to collect data.\n"
                            f"Affected ticker: {ticker}\n"
                            f"Recommendation: {recommendation}"
                        )

                        tasks.append(
                            council.run_debate(
                                cycle_id=cycle_id,
                                target_type="scraper",
                                target_name=source,
                                issue_description=issue_desc,
                            )
                        )

                # ── 3. Decision Issues → Route to Constitution Amendment Debate ──
                # Decision issues (low win rate, poor calibration, risk imbalance) are
                # now actionable because scoring uses actual trade outcomes.
                # Route critical/warning issues to the Debate Council as constitution
                # amendments (adjusting thresholds, position sizing, etc.).
                actionable_decision_issues = [
                    i for i in decision_issues
                    if i.get("severity") in ("critical", "warning")
                ]
                if actionable_decision_issues:
                    # Circuit breaker check for decision evolution
                    if self._is_circuit_open("constitution_amendment", "decision_quality"):
                        skipped_circuit.append("constitution_amendment:decision_quality")
                    else:
                        issue_desc = (
                            "Decision quality issues detected from trade outcome analysis:\n"
                            + "\n".join(
                                f"- [{i.get('severity', 'info').upper()}] {i.get('issue', '')}"
                                for i in actionable_decision_issues[:5]
                            )
                            + "\n\nPropose a constitution parameter adjustment "
                            "(e.g., max_positions, rsi_threshold, max_holding_days) "
                            "that would address these issues based on the outcome data."
                        )
                        tasks.append(
                            council.run_debate(
                                cycle_id=cycle_id,
                                target_type="constitution_amendment",
                                target_name="decision_quality",
                                issue_description=issue_desc,
                            )
                        )
                        logger.info(
                            "[EVO-ROUTER] Routing %d decision issues to Constitution debate",
                            len(actionable_decision_issues),
                        )
                elif decision_issues:
                    logger.info(
                        "[EVO-ROUTER] %d decision issues are info-level only, skipping debate",
                        len(decision_issues),
                    )

                # Log circuit breaker skips
                if skipped_circuit:
                    logger.info(
                        "[EVO-ROUTER] Circuit breaker tripped for %d targets "
                        "(rejected %d+ consecutive times): %s",
                        len(skipped_circuit),
                        self.MAX_CONSECUTIVE_REJECTIONS,
                        ", ".join(skipped_circuit),
                    )

                # ── Dispatch all tasks with adaptive concurrency (8-16 dynamic) ──
                if tasks:
                    logger.info(
                        "[EVO-ROUTER] Dispatching %d evolution debates (adaptive concurrency)...",
                        len(tasks),
                    )
                    from app.services.adaptive_concurrency import concurrency_controller
                    results = await concurrency_controller.gather(
                        tasks, label="evolution_router"
                    )
                    logger.info("[EVO-ROUTER] All evolution debates complete.")

                    # Auto-deploy approved fixes
                    from app.cognition.evolution.deployer import deploy_fix_to_disk

                    for res in results:
                        if (
                            isinstance(res, dict)
                            and res.get("status") == "pending"
                            and res.get("fix_id")
                        ):
                            fix_id = res["fix_id"]
                            logger.info(
                                "[EVO-ROUTER] Auto-deploying approved fix %s", fix_id
                            )
                            deploy_res = deploy_fix_to_disk(fix_id)
                            if "error" in deploy_res:
                                logger.error(
                                    "[EVO-ROUTER] Auto-deploy failed for %s: %s",
                                    fix_id,
                                    deploy_res["error"],
                                )
                            else:
                                logger.info(
                                    "[EVO-ROUTER] Auto-deploy successful for %s", fix_id
                                )
                else:
                    logger.info(
                        "[EVO-ROUTER] No actionable issues found — skipping evolution."
                    )

            except Exception as e:
                logger.error("[EVO-ROUTER] Failed to run router: %s", e)


async def request_evolution_debate(
    category: str,
    issue_description: str,
    error_trace: str = "",
    severity: str = "medium",
) -> dict:
    """Manually request an evolution debate (e.g. from a tool call).

    This skips the AutoResearch reporting pipeline and directly
    queues a debate. Useful for autonomous agents (like the Benchmark Agent)
    that want to propose rule amendments directly.
    """
    logger.info(
        "[EVO-ROUTER] Direct debate request received for category: %s", category
    )

    # Generate a unique cycle ID for this on-demand debate
    import uuid
    import datetime

    on_demand_cycle = f"ondemand_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}"

    try:
        # We target the 'constitution_amendment' type here, but the target_name
        # could be the rule_id if we want the mapping engine to handle it,
        # or we just use it directly.
        # Since 'constitution_amendment' isn't a python file, we handle it as a special target_type.
        target_name = "trading_constitution"

        full_issue = f"{issue_description}\n\nDetails:\n{error_trace}"

        result = await council.run_debate(
            cycle_id=on_demand_cycle,
            target_type=category,
            target_name=target_name,
            issue_description=full_issue,
        )

        # If it was approved, we deploy the Constitution amendment immediately
        # (This differs from code fixes, which go to disk. Constitution fixes go to DB)
        if (
            isinstance(result, dict)
            and result.get("status") == "pending"
            and result.get("fix_id")
        ):
            fix_id = result["fix_id"]
            if category == "constitution_amendment":
                _deploy_constitution_amendment(fix_id, error_trace)
                return {
                    "status": "success",
                    "message": "Amendment approved and deployed.",
                    "fix_id": fix_id,
                }
            else:
                from app.cognition.evolution.deployer import deploy_fix_to_disk

                deploy_res = deploy_fix_to_disk(fix_id)
                return {
                    "status": "success",
                    "message": "Debate completed and fix deployed.",
                    "deploy_res": deploy_res,
                }

        return {
            "status": "success",
            "message": "Debate completed but fix was rejected or not pending.",
        }
    except Exception as e:
        logger.error("[EVO-ROUTER] Direct debate failed: %s", e)
        return {"status": "error", "message": str(e)}


def _deploy_constitution_amendment(fix_id: str, payload_json: str):
    """Deploy an approved Constitution amendment directly to the DB."""
    try:
        payload = json.loads(payload_json)
        rule_id = payload.get("rule_id")
        param_name = payload.get("param_name")
        proposed_value = payload.get("proposed_value")
        rationale = payload.get("rationale")

        if not all([rule_id, param_name, proposed_value]):
            logger.error("[EVO-ROUTER] Missing amendment payload data.")
            return

        with get_db() as db:
            # We simply update the rule params inline here for simplicity.
            # Real versioning would create a new row and disable the old one.
            row = db.execute(
                "SELECT rule_params FROM trading_constitution WHERE id = %s", [rule_id]
            ).fetchone()
            if not row:
                return

            params = json.loads(
                row[0] if isinstance(row[0], str) else json.dumps(row[0])
            )
            params[param_name] = proposed_value

            db.execute(
                "UPDATE trading_constitution SET rule_params = %s, amended_at = CURRENT_TIMESTAMP, amendment_reason = %s WHERE id = %s",
                [json.dumps(params), rationale, rule_id],
            )
            db.commit()
            logger.info(
                "[EVO-ROUTER] Applied constitution amendment to %s: %s = %s",
                rule_id,
                param_name,
                proposed_value,
            )
    except Exception as e:
        logger.error("[EVO-ROUTER] Failed to deploy constitution amendment: %s", e)


router = EvolutionRouter()
