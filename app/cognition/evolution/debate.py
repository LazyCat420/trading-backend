"""
Evolution Debate Council — Distributes evolutionary fixes across all 3 hardware endpoints.

Implements a Peer-Review Council (Proposer, Critic, Judge) to prevent model bias
when autonomously fixing prompts, scrapers, and strategies.

Now integrates:
  - Target Mapping Engine: resolves issue names → actual file paths + contents
  - Test & Prove Environment: validates proposed fixes before the Judge rules
"""

import ast
import json
import logging
import random
import re
import traceback
import uuid
from typing import Optional

from app.db.connection import get_db
from app.services.vllm_client import llm, Priority
from app.services.prism_agent_caller import call_prism_agent

logger = logging.getLogger(__name__)


class EvolutionDebateCouncil:
    def __init__(self):
        # The primary hardware endpoints
        self.endpoints = ["jetson", "dgx_spark"]

    def _get_rotation_schedule(self) -> list[dict[str, str]]:
        """Build a deterministic 3-round rotation across available endpoints.
        Since only 2 endpoints are available, we rotate roles using fallback duplication.
        """
        return [
            {"proposer": "jetson", "critic": "dgx_spark", "judge": "dgx_spark"},
            {"proposer": "dgx_spark", "critic": "jetson", "judge": "dgx_spark"},
            {"proposer": "dgx_spark", "critic": "dgx_spark", "judge": "jetson"},
        ]

    async def run_debate(
        self,
        cycle_id: str,
        target_type: str,
        target_name: str,
        issue_description: str,
        current_content: str | None = None,
    ) -> Optional[str]:
        """
        Runs a 3-round debate where each box plays every role exactly once.
        The final decision uses majority vote (2/3 APPROVE needed).
        The best-scoring proposal (from the approved rounds) is selected.

        Returns the accepted fix (str) or None if rejected.
        """
        rotation = self._get_rotation_schedule()
        logger.info(
            "[EVO-DEBATE] Starting 3-round Council for %s '%s'",
            target_type,
            target_name,
        )
        for i, r in enumerate(rotation):
            logger.info(
                "[EVO-DEBATE]   Round %d: Proposer=%s  Critic=%s  Judge=%s",
                i + 1,
                r["proposer"],
                r["critic"],
                r["judge"],
            )

        # ── 0. RESOLVE: Use the Mapping Engine to find the real file ──
        resolved_content = current_content
        resolved_path = None
        if not resolved_content:
            try:
                from app.cognition.evolution.target_map import resolve_target

                target_info = resolve_target(target_type, target_name)
                resolved_content = (
                    target_info.get("content") or "# File not found or empty"
                )
                resolved_path = target_info.get("relative_path")
                logger.info(
                    "[EVO-DEBATE] Resolved %s -> %s (exists=%s)",
                    target_name,
                    resolved_path,
                    target_info.get("exists"),
                )
            except Exception as e:
                logger.error("[EVO-DEBATE] Target mapping failed: %s", e)
                resolved_content = f"# Could not resolve target: {e}"

        # ── 0b. STABLE FALLBACK: prefer last known-good version if disk is broken ──
        try:
            from app.cognition.evolution.deployer import get_stable_version

            stable = get_stable_version(target_type, target_name)
            if stable:
                # If disk content is missing/broken, use the stable version
                disk_broken = (
                    not resolved_content
                    or resolved_content.startswith("# File not found")
                    or resolved_content.startswith("# Could not resolve")
                    or len(resolved_content.strip()) < 50
                )
                if disk_broken:
                    logger.info(
                        "[EVO-DEBATE] Disk content broken — using STABLE version for %s/%s (%d chars)",
                        target_type, target_name, len(stable),
                    )
                    resolved_content = stable
                else:
                    # Even if disk is OK, append stable version as reference
                    logger.debug(
                        "[EVO-DEBATE] Stable version available for %s/%s (using current disk)",
                        target_type, target_name,
                    )
        except Exception as stbl_err:
            logger.debug("[EVO-DEBATE] Stable fallback lookup failed: %s", stbl_err)

        # ── Evidence Gathering (Phase 1: online research) ──
        evidence_context = ""
        try:
            evidence_context = await self._gather_evidence(
                target_type, target_name, issue_description
            )
            if evidence_context:
                logger.info(
                    "[EVO-DEBATE] Gathered %d chars of online evidence",
                    len(evidence_context),
                )
        except Exception as ev_err:
            logger.warning("[EVO-DEBATE] Evidence gathering failed: %s", ev_err)

        # ── Dead-End Memory (Phase 4: don't repeat failed approaches) ──
        dead_end_context = ""
        try:
            dead_end_context = self._get_dead_ends(target_type, target_name)
        except Exception as de_err:
            logger.debug("[EVO-DEBATE] Dead-end lookup failed: %s", de_err)

        # ── Run 3 rounds ──
        round_results: list[dict] = []

        for round_num, roles in enumerate(rotation, 1):
            logger.info("[EVO-DEBATE] ── Round %d / 3 ──", round_num)

            # Feed rejection feedback from previous rounds into this round
            prev_feedback = ""
            if round_results:
                rejected_rounds = [r for r in round_results if r.get("decision") == "REJECT"]
                if rejected_rounds:
                    feedback_parts = []
                    for r in rejected_rounds:
                        fb = r.get("rejection_feedback", "")
                        concerns = r.get("critic_concerns", "")
                        if fb:
                            feedback_parts.append(f"Round {r['round']} Judge: {fb}")
                        if concerns:
                            feedback_parts.append(f"Round {r['round']} Critic: {concerns[:300]}")
                    if feedback_parts:
                        prev_feedback = (
                            "\n── PREVIOUS ROUND FEEDBACK (address these specific concerns) ──\n"
                            + "\n".join(feedback_parts[:4])
                        )

            result = await self._run_single_round(
                round_num=round_num,
                roles=roles,
                cycle_id=cycle_id,
                target_type=target_type,
                target_name=target_name,
                issue_description=issue_description,
                resolved_content=resolved_content,
                resolved_path=resolved_path,
                evidence_context=evidence_context,
                dead_end_context=dead_end_context,
                prev_round_feedback=prev_feedback,
            )
            round_results.append(result)

        # ── Majority Vote ──
        approvals = [r for r in round_results if r["decision"] == "APPROVE"]
        rejections = [r for r in round_results if r["decision"] == "REJECT"]

        logger.info(
            "[EVO-DEBATE] Majority vote: %d APPROVE / %d REJECT",
            len(approvals),
            len(rejections),
        )

        if len(approvals) >= 2:
            # Pick the best-scoring approved proposal
            best = max(approvals, key=lambda r: r.get("judge_score", 0))
            final_status = "pending"
            final_fix = best["proposed_fix"]
            logger.info(
                "[EVO-DEBATE] APPROVED by majority (best from round %d)", best["round"]
            )
        else:
            best = round_results[0]
            final_status = "rejected"
            final_fix = None
            logger.info("[EVO-DEBATE] REJECTED by majority")

        # Persist the winning (or losing) result
        fix_id = self._save_pending_fix(
            cycle_id=cycle_id,
            target_type=target_type,
            target_name=target_name,
            file_path=resolved_path,
            proposed_fix=best.get("proposed_fix", ""),
            all_rounds=round_results,
            judge_score=1.0 if final_status == "pending" else 0.0,
            status=final_status,
        )

        return {"fix": final_fix, "fix_id": fix_id, "status": final_status}

    async def _run_single_round(
        self,
        round_num: int,
        roles: dict[str, str],
        cycle_id: str,
        target_type: str,
        target_name: str,
        issue_description: str,
        resolved_content: str,
        resolved_path: str | None,
        evidence_context: str = "",
        dead_end_context: str = "",
        prev_round_feedback: str = "",
    ) -> dict:
        """Execute one Proposer→Critic→Test→Judge round. Returns a result dict."""

        round_tag = f"R{round_num}"

        # ── 1. PROPOSER ──
        precise_context = self._extract_relevant_context(
            resolved_content, issue_description
        )

        proposer_system = (
            f"You are the Lead Engineer for an autonomous trading system.\n"
            f"You need to fix a failing {target_type} "
            f"{'at ' + resolved_path if resolved_path else f'named {target_name}'}.\n"
            f"Analyze the issue and propose a COMPLETE rewrite that fixes the problem.\n"
            f"State: (1) what is broken, (2) why your fix works, "
            f"(3) what improvement you predict.\n"
            f"Output your reasoning, then the complete rewritten content "
            f"wrapped in ```python``` blocks (or ```markdown``` for .md files)."
        )

        # Build enriched user prompt with evidence + dead-ends + feedback
        user_parts = [f"ISSUE DESCRIPTION:\n{issue_description}"]
        if prev_round_feedback:
            user_parts.append(prev_round_feedback)
        if dead_end_context:
            user_parts.append(f"\n{dead_end_context}")
        if evidence_context:
            user_parts.append(f"\n{evidence_context}")
        user_parts.append(
            f"\nCURRENT FILE CONTENT (extracted relevant context):\n"
            f"```\n{precise_context}\n```\n\n"
            f"Draft your proposed fix. Wrap the COMPLETE rewritten code "
            f"in a single ```python``` fenced block."
        )
        proposer_user = "\n".join(user_parts)

        try:
            proposed_response, _, _ = await call_prism_agent(
                agent_id="CUSTOM_EVO_PROPOSER_AGENT",
                user_message=proposer_user,
                fallback_system_prompt=proposer_system,
                fallback_agent_name=f"evo_proposer_{round_tag}",
                temperature=0.4,
                max_tokens=4096,
                priority=Priority.LOW,
                cycle_id=cycle_id,
            )
        except Exception as e:
            tb = traceback.format_exc()
            err_msg = f"{str(e) or repr(e)}\n{tb}"
            logger.error(
                "[EVO-DEBATE] %s Proposer (%s) failed:\n%s",
                round_tag,
                roles["proposer"],
                err_msg,
            )
            return {
                "round": round_num,
                "decision": "REJECT",
                "error": err_msg,
                "roles": roles,
            }

        # Phase 2: Format Normalizer — clean before extracting
        proposed_fix = self._extract_code(self._normalize_proposal(proposed_response))
        if not proposed_fix:
            logger.warning(
                "[EVO-DEBATE] %s Proposer produced no code block.", round_tag
            )
            return {
                "round": round_num,
                "decision": "REJECT",
                "error": "No code block after normalization",
                "roles": roles,
                "proposer_rationale": proposed_response[:2000],
            }

        # ── 2. CRITIC (Cross-Verification) ──
        critic_system = (
            f"You are a Red Team Security Auditor and Claim Verifier.\n"
            f"Review the Proposer's fix for {target_type} "
            f"{'at ' + resolved_path if resolved_path else target_name}.\n"
            f"For each claim the Proposer makes, classify it as:\n"
            f"  VERIFIED — evidence supports it\n"
            f"  UNVERIFIED — no evidence found\n"
            f"  CONTRADICTED — evidence disagrees\n"
            f"Also check for: hallucinated APIs, logic bugs, edge cases.\n"
            f"If ≥50% of claims are UNVERIFIED or CONTRADICTED, flag HALLUCINATION.\n"
            f"End with: 'NO_CONCERNS' if all checks pass, or list concerns."
        )
        critic_parts = [
            f"ISSUE:\n{issue_description}",
            f"ORIGINAL:\n```\n{precise_context}\n```",
            f"PROPOSED FIX:\n```\n{proposed_fix}\n```",
            f"PROPOSER RATIONALE:\n{proposed_response[:2000]}",
        ]
        if evidence_context:
            critic_parts.append(
                f"INDEPENDENT EVIDENCE (use to verify Proposer claims):\n{evidence_context}"
            )
        critic_parts.append("Verify the Proposer's claims and list concerns.")
        critic_user = "\n\n".join(critic_parts)

        try:
            critic_response, _, _ = await call_prism_agent(
                agent_id="CUSTOM_EVO_CRITIC_AGENT",
                user_message=critic_user,
                fallback_system_prompt=critic_system,
                fallback_agent_name=f"evo_critic_{round_tag}",
                temperature=0.3,
                max_tokens=2048,
                priority=Priority.LOW,
                cycle_id=cycle_id,
            )
        except Exception as e:
            tb = traceback.format_exc()
            err_msg = f"{str(e) or repr(e)}\n{tb}"
            logger.error(
                "[EVO-DEBATE] %s Critic (%s) failed:\n%s",
                round_tag,
                roles["critic"],
                err_msg,
            )
            return {
                "round": round_num,
                "decision": "REJECT",
                "error": err_msg,
                "roles": roles,
                "proposed_fix": proposed_fix,
            }

        # ── 3. TEST & PROVE ──
        test_result = {"passed": True, "details": "No tests available", "errors": []}
        try:
            from app.cognition.evolution.test_prove import validate_fix

            test_result = await validate_fix(
                target_type=target_type,
                target_name=target_name,
                proposed_fix=proposed_fix,
                issue_description=issue_description,
            )
            logger.info(
                "[EVO-DEBATE] %s Test: passed=%s (%d/%d)",
                round_tag,
                test_result["passed"],
                test_result.get("tests_passed", 0),
                test_result.get("tests_run", 0),
            )
        except Exception as e:
            logger.error("[EVO-DEBATE] %s Test crashed: %s", round_tag, e)
            test_result = {
                "passed": False,
                "details": f"Crashed: {e}",
                "errors": [str(e)],
            }

        # ── 4. JUDGE ──
        test_summary = (
            f"AUTOMATED TEST RESULTS:\n"
            f"  Passed: {'YES' if test_result['passed'] else 'NO'}\n"
            f"  Details: {test_result['details']}\n"
        )
        if test_result.get("errors"):
            test_summary += f"  Errors: {'; '.join(test_result['errors'])}\n"

        judge_system = (
            "You are the Supreme Arbiter of an autonomous AI pipeline.\n"
            "Review the fix, the Critic's concerns, AND the automated test results.\n"
            "If automated tests FAILED due to SYNTAX ERRORS, you MUST reject.\n"
            "If tests flagged minor style or import issues, use your judgment — \n"
            "scrapers legitimately use subprocess, os, and external tools.\n\n"
            "Score the proposal on these 3 dimensions (0-100 each):\n"
            "  CODE_QUALITY: syntax correctness, style, maintainability (0-100)\n"
            "  ISSUE_RESOLUTION: does this actually fix the reported issue? (0-100)\n"
            "  SIDE_EFFECT_RISK: risk of breaking existing functionality, lower=safer (0-100, 0=no risk, 100=very risky)\n\n"
            "Output format (EXACTLY):\n"
            "CODE_QUALITY: <score>\n"
            "ISSUE_RESOLUTION: <score>\n"
            "SIDE_EFFECT_RISK: <score>\n"
            "REASONING: <1-2 sentences explaining your key concern or endorsement>\n"
            "VERDICT: APPROVE or REJECT"
        )
        judge_user = (
            f"PROPOSED FIX:\n```\n{proposed_fix[:3000]}\n```\n\n"
            f"CRITIC CONCERNS:\n{critic_response[:2000]}\n\n"
            f"{test_summary}\nMake your ruling."
        )

        try:
            judge_response, _, _ = await call_prism_agent(
                agent_id="CUSTOM_EVO_JUDGE_AGENT",
                user_message=judge_user,
                fallback_system_prompt=judge_system,
                fallback_agent_name=f"evo_judge_{round_tag}",
                temperature=0.1,
                max_tokens=2048,
                priority=Priority.LOW,
                cycle_id=cycle_id,
            )
        except Exception as e:
            tb = traceback.format_exc()
            err_msg = f"{str(e) or repr(e)}\n{tb}"
            logger.error(
                "[EVO-DEBATE] %s Judge (%s) failed:\n%s",
                round_tag,
                roles["judge"],
                err_msg,
            )
            return {
                "round": round_num,
                "decision": "REJECT",
                "error": err_msg,
                "roles": roles,
                "proposed_fix": proposed_fix,
            }

        # Parse decision and scores
        decision = "REJECT"
        judge_scores = {"code_quality": 50, "issue_resolution": 50, "side_effect_risk": 50}
        rejection_feedback = ""

        for line in judge_response.strip().split("\n"):
            stripped = line.strip()
            upper = stripped.upper()

            # Parse dimension scores
            if upper.startswith("CODE_QUALITY:"):
                try:
                    judge_scores["code_quality"] = int(re.search(r'(\d+)', stripped).group(1))
                except (ValueError, AttributeError):
                    pass
            elif upper.startswith("ISSUE_RESOLUTION:"):
                try:
                    judge_scores["issue_resolution"] = int(re.search(r'(\d+)', stripped).group(1))
                except (ValueError, AttributeError):
                    pass
            elif upper.startswith("SIDE_EFFECT_RISK:"):
                try:
                    judge_scores["side_effect_risk"] = int(re.search(r'(\d+)', stripped).group(1))
                except (ValueError, AttributeError):
                    pass
            elif upper.startswith("REASONING:"):
                rejection_feedback = stripped[len("REASONING:"):].strip()

        # Parse the APPROVE/REJECT verdict
        for line in reversed(judge_response.strip().split("\n")):
            stripped = line.strip().upper()
            if "APPROVE" in stripped and "VERDICT" in stripped or stripped == "APPROVE":
                decision = "APPROVE"
                break
            elif "REJECT" in stripped and "VERDICT" in stripped or stripped == "REJECT":
                decision = "REJECT"
                break
            elif "APPROVE" in stripped:
                decision = "APPROVE"
                break
            elif "REJECT" in stripped:
                decision = "REJECT"
                break

        # Hard override: syntax errors = REJECT (but not minor test flags)
        critical_errors = [
            e
            for e in test_result.get("errors", [])
            if "syntax" in e.lower() or "Syntax" in e
        ]
        if critical_errors and decision == "APPROVE":
            logger.warning(
                "[EVO-DEBATE] %s Judge approved but SYNTAX tests FAILED — forcing REJECT",
                round_tag,
            )
            decision = "REJECT"
            rejection_feedback = f"Syntax errors detected: {'; '.join(critical_errors[:3])}"

        # Compute granular judge_score (0.0 - 1.0)
        # Formula: (code_quality * 0.3 + issue_resolution * 0.4 + (100 - side_effect_risk) * 0.3) / 100
        cq = min(100, max(0, judge_scores["code_quality"]))
        ir = min(100, max(0, judge_scores["issue_resolution"]))
        ser = min(100, max(0, judge_scores["side_effect_risk"]))
        judge_score = (cq * 0.3 + ir * 0.4 + (100 - ser) * 0.3) / 100

        # Auto-threshold: if score >= 0.6 and verdict was ambiguous, approve
        if judge_score >= 0.6 and decision == "REJECT" and not critical_errors:
            # Only override if tests passed and score is strong
            if test_result.get("passed", False) and judge_score >= 0.75:
                logger.info(
                    "[EVO-DEBATE] %s Score-based override: %.2f >= 0.75 with passing tests → APPROVE",
                    round_tag, judge_score,
                )
                decision = "APPROVE"

        logger.info(
            "[EVO-DEBATE] %s Decision: %s (score=%.2f, cq=%d ir=%d ser=%d)",
            round_tag,
            decision,
            judge_score,
            cq, ir, ser,
        )

        return {
            "round": round_num,
            "decision": decision,
            "judge_score": round(judge_score, 3),
            "judge_scores": judge_scores,
            "rejection_feedback": rejection_feedback if decision == "REJECT" else "",
            "proposed_fix": proposed_fix,
            "proposer_rationale": proposed_response[:2000],
            "critic_concerns": critic_response[:2000],
            "judge_reasoning": judge_response[:2000],
            "test_result": test_result,
            "roles": roles,
        }

    def _extract_code(self, text: str) -> str:
        """Extract code from markdown fences with multiple fallback strategies."""
        # Strategy 1: Standard fenced code block (```python\n...```)
        match = re.search(r"```(?:\w+)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Strategy 2: Fenced block with code on same line (```python code...```)
        match = re.search(r"```(?:\w+)?\s+(.+?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Strategy 3: Single unclosed fence — take everything after it
        match = re.search(r"```(?:\w+)?\s*\n(.+)", text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Remove trailing ``` if present
            if code.endswith("```"):
                code = code[:-3].strip()
            if len(code) > 50:  # Only accept if substantial
                return code

    def _extract_relevant_context(self, content: str, issue_description: str) -> str:
        """Extract precise context (classes/functions) mentioned in the issue using AST."""
        if len(content) <= 4000:
            return content

        # Extract potential identifiers from the issue description
        keywords = [
            word.lower()
            for word in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]+\b", issue_description)
            if len(word) > 3
        ]

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return content[:4000] + "\n... [TRUNCATED DUE TO SYNTAX ERROR]"

        relevant_nodes = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if any(kw in node.name.lower() for kw in keywords):
                    relevant_nodes.append(node)

        if relevant_nodes:
            lines = content.splitlines()
            extracted = []
            for node in relevant_nodes:
                start = node.lineno - 1
                end = getattr(node, "end_lineno", start + 15)
                extracted.append("\n".join(lines[start:end]))

            combined = "\n\n...\n\n".join(extracted)
            if len(combined) > 4000:
                return combined[:4000] + "\n... [TRUNCATED]"
            return combined

        return content[:4000] + "\n... [TRUNCATED]"

    def _save_pending_fix(
        self,
        cycle_id: str,
        target_type: str,
        target_name: str,
        file_path: str | None,
        proposed_fix: str,
        all_rounds: list[dict],
        judge_score: float,
        status: str,
    ):
        """Save the 3-round debate result to the pending_evolution_fixes table."""
        with get_db() as db:
            # Build a rich metadata blob from all 3 rounds
            rounds_summary = []
            for r in all_rounds:
                rounds_summary.append(
                    {
                        "round": r.get("round"),
                        "decision": r.get("decision"),
                        "score": r.get("judge_score", 0),
                        "judge_scores": r.get("judge_scores", {}),
                        "rejection_feedback": r.get("rejection_feedback", ""),
                        "roles": r.get("roles", {}),
                        "proposer_rationale": r.get("proposer_rationale", "")[:500],
                        "critic_concerns": r.get("critic_concerns", "")[:500],
                        "judge_reasoning": r.get("judge_reasoning", "")[:500],
                        "test_passed": r.get("test_result", {}).get("passed"),
                        "test_details": r.get("test_result", {}).get("details", ""),
                        "error": r.get("error"),
                    }
                )

            # Collect structured rejection reasons for dead-end memory
            rejection_reasons = [
                r.get("rejection_feedback", "")
                for r in all_rounds
                if r.get("decision") == "REJECT" and r.get("rejection_feedback")
            ]

            motivation_blob = json.dumps(
                {
                    "file_path": file_path,
                    "rounds": rounds_summary,
                    "vote": f"{sum(1 for r in all_rounds if r['decision'] == 'APPROVE')}/3 APPROVE",
                    "rejection_reasons": rejection_reasons,
                }
            )

            # Combine critic concerns from all rounds
            critic_blob = "\n---\n".join(
                f"Round {r.get('round')}: {r.get('critic_concerns', 'N/A')[:600]}"
                for r in all_rounds
                if r.get("critic_concerns")
            )

            # Proposer model = the winning round's proposer endpoint
            winning_round = next(
                (r for r in all_rounds if r.get("decision") == "APPROVE"),
                all_rounds[0] if all_rounds else {},
            )
            proposer_model = winning_round.get("roles", {}).get("proposer", "unknown")

            fix_id = str(uuid.uuid4())

            try:
                db.execute(
                    """
                    INSERT INTO pending_evolution_fixes (
                        id, cycle_id, target_type, target_name, proposed_fix,
                        motivation, proposer_model, critic_concerns, judge_score, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        fix_id,
                        cycle_id,
                        target_type,
                        target_name,
                        proposed_fix,
                        motivation_blob,
                        proposer_model,
                        critic_blob[:3000],
                        judge_score,
                        status,
                    ],
                )
                logger.info(
                    "[EVO-DEBATE] Saved %s fix for %s (vote=%s)",
                    status,
                    target_name,
                    f"{sum(1 for r in all_rounds if r['decision'] == 'APPROVE')}/3",
                )
                return fix_id
            except Exception as e:
                logger.error("[EVO-DEBATE] Failed to save pending fix: %s", e)
                return None

    # ═══════════════════════════════════════════════════════════════
    # EVIDENCE GATHERING — Online research for Proposer/Critic
    # ═══════════════════════════════════════════════════════════════

    async def _gather_evidence(
        self, target_type: str, target_name: str, issue_description: str
    ) -> str:
        """Two-tier evidence gathering for the debate council.

        Tier 1: DuckDuckGo text search — fast, factual snippets (docs, SO, etc.)
        Tier 2: Hermes gateway — deeper LLM-synthesized research via the 3 vLLM
                boxes. Only fires if DDG returns thin results (< 2 hits).

        Safety: DDG is read-only. Hermes goes through guardrails
        (validate_hermes_request + sanitize_hermes_output) which block
        dangerous prompts, package installs, and command execution.
        """
        evidence_parts: list[str] = []

        # ── Tier 1: DuckDuckGo snippets (fast, free) ──
        ddg_results = []
        try:
            from app.services.web_search import searcher

            short_issue = issue_description[:120].replace("\n", " ")
            query = f"python {target_type} {target_name} fix: {short_issue}"
            ddg_results = await searcher.search(query, max_results=3, timelimit=None)

            if ddg_results:
                evidence_parts.append("── DDG SEARCH RESULTS ──")
                for i, r in enumerate(ddg_results, 1):
                    evidence_parts.append(
                        f"[{i}] {r.title}\n    URL: {r.url}\n    {r.snippet[:500]}"
                    )
                logger.info("[EVO-DEBATE] DDG evidence: %d results", len(ddg_results))
        except Exception as ddg_err:
            logger.warning("[EVO-DEBATE] DDG search failed: %s", ddg_err)

        # ── Tier 2: Hermes deep research (if DDG was thin) ──
        # Guardrails: 30s hard timeout, repetition detection, mock-response filter
        if len(ddg_results) < 2:
            try:
                import asyncio as _aio
                from app.tools.web_tools import hermes_web_research

                hermes_query = (
                    f"Research how to fix a {target_type} issue in an autonomous "
                    f"trading system. Target: {target_name}. "
                    f"Issue: {issue_description[:200]}. "
                    f"Provide specific Python code patterns and best practices."
                )

                # Hard 30s timeout — evidence gathering should be fast,
                # not block the debate for 10 minutes if Hermes is stuck
                hermes_raw = await _aio.wait_for(
                    hermes_web_research(hermes_query), timeout=30.0
                )

                # Parse the JSON response from hermes
                import json as _json

                hermes_data = _json.loads(hermes_raw)
                hermes_text = hermes_data.get("response", "")
                hermes_status = hermes_data.get("status", "")

                # Filter: skip mock/empty responses
                if hermes_status == "mock_success" or not hermes_text:
                    logger.info("[EVO-DEBATE] Hermes returned mock/empty — skipping")
                elif hermes_status == "blocked":
                    logger.info(
                        "[EVO-DEBATE] Hermes blocked by guardrails: %s",
                        hermes_data.get("reason", "unknown"),
                    )
                elif hermes_status == "success":
                    # Doom-loop detection: check for excessive repetition
                    # If >40% of sentences repeat, Hermes is stuck
                    sentences = [
                        s.strip() for s in hermes_text.split(".") if len(s.strip()) > 20
                    ]
                    unique = set(sentences)
                    if sentences and len(unique) / len(sentences) < 0.6:
                        logger.warning(
                            "[EVO-DEBATE] Hermes doom-loop detected: %d/%d unique "
                            "sentences — discarding output",
                            len(unique),
                            len(sentences),
                        )
                    else:
                        evidence_parts.append("\n── HERMES DEEP RESEARCH ──")
                        evidence_parts.append(hermes_text[:2000])
                        logger.info(
                            "[EVO-DEBATE] Hermes evidence: %d chars", len(hermes_text)
                        )

            except _aio.TimeoutError:
                logger.warning(
                    "[EVO-DEBATE] Hermes evidence timed out after 30s — skipping"
                )
            except Exception as hermes_err:
                logger.warning("[EVO-DEBATE] Hermes research failed: %s", hermes_err)

        if not evidence_parts:
            return ""

        full_evidence = (
            "── ONLINE EVIDENCE (read-only, cite sources) ──\n"
            + "\n\n".join(evidence_parts)
        )
        return full_evidence[:4000]

    # ═══════════════════════════════════════════════════════════════
    # FORMAT NORMALIZER — Fix blank proposals
    # ═══════════════════════════════════════════════════════════════

    def _normalize_proposal(self, raw_response: str) -> str:
        """Clean LLM response before code extraction.

        Handles: missing fences, conversational preamble, unclosed blocks,
        and responses that are pure code without markdown wrapping.
        """
        if not raw_response:
            return ""

        text = raw_response.strip()

        # If it already has proper code fences, return as-is
        if "```" in text:
            return text

        # Try to detect if the entire response IS code (no markdown)
        lines = text.splitlines()
        code_indicators = 0
        for line in lines[:20]:
            stripped = line.strip()
            if stripped.startswith(
                ("import ", "from ", "def ", "class ", "async def ", "    ", "@", "#!")
            ):
                code_indicators += 1

        # If >40% of first 20 lines look like code, wrap it
        if lines and code_indicators / max(len(lines[:20]), 1) > 0.4:
            return f"```python\n{text}\n```"

        # Strip common conversational preamble
        preamble_patterns = [
            r"^(?:Here(?:'s| is) (?:the |my )?(?:proposed |updated )?(?:fix|code|solution|rewrite)[:\.]?\s*\n?)",
            r"^(?:I (?:would |will )?(?:suggest|propose|recommend)[^.]*\.\s*\n?)",
            r"^(?:(?:The )?(?:issue|problem|bug) (?:is|was)[^.]*\.\s*\n?)",
        ]
        for pattern in preamble_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

        return text.strip()

    # ═══════════════════════════════════════════════════════════════
    # DEAD-END MEMORY — Prevent repeating failed approaches
    # ═══════════════════════════════════════════════════════════════

    def _get_dead_ends(self, target_type: str, target_name: str) -> str:
        """Query evolution_dead_ends for this target. Returns context string."""
        try:
            with get_db() as db:
                rows = db.execute(
                    "SELECT failure_reason, created_at FROM evolution_dead_ends "
                    "WHERE target_type = %s AND target_name = %s "
                    "ORDER BY created_at DESC LIMIT 3",
                    [target_type, target_name],
                ).fetchall()

                if not rows:
                    return ""

                context = (
                    "── DEAD-END MEMORY (approaches that FAILED — do NOT repeat) ──\n"
                )
                for i, row in enumerate(rows, 1):
                    context += f"{i}. {row[0]}\n"

                return context[:2000]
        except Exception:
            return ""


council = EvolutionDebateCouncil()
