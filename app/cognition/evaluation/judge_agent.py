import asyncio
import json
import logging
import os
from enum import Enum

from app.services.vllm_client import llm, Priority
from app.utils.text_utils import (
    parse_json_response,
    extract_reasoning_text,
    normalize_for_rouge,
    compute_citation_overlap,
)
from app.db.connection import get_db

from .oracle import DataCompletenessOracle

# Grounding score weights: ROUGE-L precision (textual overlap) vs citation
# overlap (numeric data point grounding).  Citation is weighted higher because
# it directly measures whether the bot references actual data values from the
# context.  Increase ROUGE_WEIGHT if the LLM style becomes more verbose.
ROUGE_WEIGHT = 0.4
CITATION_WEIGHT = 0.6

# ── Failure classification codes ──
# Stored in evidence_gathering["failure_reason"] so the UI/strategy auditor
# can group low scores by root cause instead of just by number.


class FailureReason(Enum):
    NONE = "none"
    PARSE = "parse_failure"
    MISSING_CONTEXT = "missing_context"
    UNSUPPORTED_ASSET = "unsupported_asset"
    FAITHFULNESS = "faithfulness_failure"
    RELEVANCY = "relevancy_failure"
    DEEPEVAL_ERROR = "deepeval_error"
    EMPTY_RESPONSE = "empty_response"


# ── DeepEval metric thresholds (override via env or Settings) ──
FAITHFULNESS_THRESHOLD = float(os.environ.get("FAITHFULNESS_THRESHOLD", "0.7"))
RELEVANCY_THRESHOLD = float(os.environ.get("RELEVANCY_THRESHOLD", "0.5"))
# Max seconds to wait for a single DeepEval metric call before treating as error
DEEPEVAL_TIMEOUT_SEC = float(os.environ.get("DEEPEVAL_TIMEOUT_SEC", "180"))
# Max retries per DeepEval metric call before recording a red card
DEEPEVAL_MAX_RETRIES = int(os.environ.get("DEEPEVAL_MAX_RETRIES", "2"))
# Max concurrent DeepEval evaluations to prevent vLLM saturation
_DEEPEVAL_CONCURRENCY = int(os.environ.get("MAX_CONCURRENT_DEEPEVAL", "3"))
_deepeval_semaphore = asyncio.Semaphore(_DEEPEVAL_CONCURRENCY)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an independent, institutional Auditor Agent (LLM-as-a-Judge) for a quantitative trading firm.
Your job is to strictly evaluate the trading bot's proposed causal thesis using the First-Principles framework.
You DO NOT need to check data completeness or hallucinations, as deterministic systems handle those.
You care ONLY about grading the depth of the Causal Thesis.

### SCORING ANCHORS (1-5 Scale for Causal Thesis)
1 - Poor: Hallucinated connection, forced reasoning, or random associations.
2 - Weak: Contradicts context or anchors to bias rather than data.
3 - Adequate: Basic pattern matching without causal depth.
4 - Strong: Sound logic but misses minor elements like explicit invalidation.
5 - Excellent: Deep causal thesis supported by context, includes invalidation levels.

Return EXACTLY the following JSON format:
{
    "judge_score": <int 1-5>,
    "first_principles": "<A brief string extracting the bot's causal thesis, or explaining why it's missing>"
}
"""

USER_TEMPLATE = """### Decision ID: {decision_id}
### Asset: {ticker}

### Raw Context from Bot (What it saw):
{context}

### Bot's Raw Reasoning (What it decided):
{raw_response}

Act as the Auditor and score this decision. Output JSON only.
"""


async def evaluate_decision(decision_id: str) -> bool:
    """Run the LLM-as-a-Judge protocol on a single decision record."""
    with get_db() as db:
        failure_reason = FailureReason.NONE

        try:
            # 1. Fetch raw logs
            log = db.execute(
                "SELECT cycle_id, ticker, context_hash, raw_response, created_at "
                "FROM llm_audit_logs WHERE id = %s",
                [decision_id],
            ).fetchone()

            if not log:
                logger.error(f"Cannot evaluate {decision_id}. Log not found.")
                return False

            cycle_id, ticker, context_hash, raw_response, created_at = log

            # ── Classify empty/missing response early ──
            if not raw_response or raw_response.strip() == "":
                failure_reason = FailureReason.EMPTY_RESPONSE
                logger.warning(f"Empty response for {decision_id}.")

            # ── Classify parse failures: check if FINAL() can be extracted ──
            if failure_reason == FailureReason.NONE and raw_response:
                from app.utils.text_utils import parse_trading_decision

                parsed_decision = parse_trading_decision(raw_response)
                if not parsed_decision or "action" not in parsed_decision:
                    failure_reason = FailureReason.PARSE
                    logger.warning(
                        f"Parse failure for {decision_id}: no valid FINAL() found."
                    )

            # ── Classify unsupported asset: check for tool error markers ──
            if failure_reason == FailureReason.NONE and raw_response:
                error_markers = [
                    '"error":',
                    "'error':",
                    "No technicals for",
                    "No fundamentals for",
                    "No price data for",
                    "No data for",
                ]
                # Count how many distinct error markers appear
                error_count = sum(1 for m in error_markers if m in raw_response)
                if error_count >= 2:
                    failure_reason = FailureReason.UNSUPPORTED_ASSET
                    logger.warning(
                        f"Unsupported asset pattern for {decision_id} ({ticker}): "
                        f"{error_count} tool errors detected in response."
                    )

            # 2. Extract Context (if available in context_blobs)
            # Fallback to minimal context if blob expired
            context_blob = "Context Blob Missing"
            full_context_blob = ""
            if context_hash:
                blob = db.execute(
                    "SELECT content FROM context_blobs WHERE context_hash = %s",
                    [context_hash],
                ).fetchone()
                if blob:
                    full_context_blob = blob[0]  # full context for ROUGE grounding
                    # Build a representative truncation: take the first 800 chars
                    # (usually macro/header), then find the "Technicals" and
                    # "Fundamentals" sections and include 600 chars of each so
                    # the DeepEval faithfulness check can see ticker-specific data
                    # that the bot's reasoning references.
                    raw = blob[0]
                    parts = [raw[:800]]
                    for section_marker in (
                        "Technicals",
                        "Fundamentals",
                        "Balance Sheet",
                        "Price History",
                    ):
                        idx = raw.find(section_marker)
                        if idx >= 0:
                            parts.append(raw[idx : idx + 600])
                    context_blob = "\n...\n".join(parts)[:3000]
                else:
                    if failure_reason == FailureReason.NONE:
                        failure_reason = FailureReason.MISSING_CONTEXT
                    logger.warning(
                        f"Context blob missing for {decision_id} (hash={context_hash})"
                    )
            else:
                if failure_reason == FailureReason.NONE:
                    failure_reason = FailureReason.MISSING_CONTEXT
                logger.warning(f"No context_hash for {decision_id}")

            # 3. Deterministic Ground Truth Checks
            oracle_results = DataCompletenessOracle.verify_ground_truth(ticker)

            # ── EARLY EXIT: skip all LLM/DeepEval calls when input data is
            #    missing or broken.  Without real context + response the judge
            #    and DeepEval metrics just evaluate placeholder strings
            #    ("Context Blob Missing", "Empty Response"), wasting tokens and
            #    producing meaningless scores.
            _SKIP_LLM_FAILURES = {
                FailureReason.EMPTY_RESPONSE,
                FailureReason.MISSING_CONTEXT,
                FailureReason.UNSUPPORTED_ASSET,
                FailureReason.PARSE,
            }
            if failure_reason in _SKIP_LLM_FAILURES:
                logger.info(
                    "Skipping LLM evaluation for %s — pre-check failure: %s",
                    decision_id,
                    failure_reason,
                )
                evidence = oracle_results["checklist"].copy()
                evidence["failure_reason"] = failure_reason.value
                evidence["skipped_llm"] = True
                evidence_json = json.dumps(evidence)

                existing = db.execute(
                    "SELECT decision_id FROM decision_evaluations WHERE decision_id = %s",
                    [decision_id],
                ).fetchone()

                eval_row = [
                    0.0,  # judge_a_score
                    0.0,  # final_quality_score
                    json.dumps(
                        []
                    ),  # red_cards (none — failure is upstream, not hallucination)
                    f"Skipped: {failure_reason.value}",  # first_principles_reasoning
                    True,  # policy_understanding
                    evidence_json,  # evidence_gathering
                ]

                if existing:
                    db.execute(
                        """UPDATE decision_evaluations SET
                            judge_a_score = %s, final_quality_score = %s,
                            red_cards = %s, first_principles_reasoning = %s,
                            policy_understanding = %s, evidence_gathering = %s
                        WHERE decision_id = %s""",
                        eval_row + [decision_id],
                    )
                else:
                    db.execute(
                        """INSERT INTO decision_evaluations (
                            decision_id, cycle_id, ticker, timestamp,
                            judge_a_score, final_quality_score, red_cards,
                            first_principles_reasoning, policy_understanding,
                            discrepancy_trigger, evidence_gathering
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        [decision_id, cycle_id, ticker, created_at]
                        + eval_row[:5]
                        + [False, evidence_json],
                    )
                return True

            # 4. Construct Prompt (For subjective First-Principles Only)
            user_prompt = USER_TEMPLATE.format(
                decision_id=decision_id,
                ticker=ticker,
                context=context_blob,
                raw_response=raw_response or "Empty Response",
            )

            # 5. 3rd-Party Evaluation Checks (DeepEval wrapper — lazy import
            #    to avoid printing the DeepEval banner on every module load)
            from .deepeval_client import VLLMDeepEvalWrapper
            from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
            from deepeval.test_case import LLMTestCase

            eval_model = VLLMDeepEvalWrapper()
            faithfulness = FaithfulnessMetric(
                threshold=FAITHFULNESS_THRESHOLD, model=eval_model, include_reason=True
            )
            relevancy = AnswerRelevancyMetric(
                threshold=RELEVANCY_THRESHOLD, model=eval_model, include_reason=True
            )
            test_case = LLMTestCase(
                input=user_prompt,
                actual_output=raw_response or "Empty Response",
                retrieval_context=[context_blob],
            )

            red_cards = []

            # ── Faithfulness check (with retry + semaphore) ──
            faith_succeeded = False
            for attempt in range(DEEPEVAL_MAX_RETRIES):
                try:
                    async with _deepeval_semaphore:
                        await asyncio.wait_for(
                            faithfulness.a_measure(test_case),
                            timeout=DEEPEVAL_TIMEOUT_SEC,
                        )
                    logger.debug(
                        "DeepEval faithfulness for %s: score=%.3f reason=%s",
                        decision_id,
                        faithfulness.score or 0,
                        (faithfulness.reason or "")[:200],
                    )
                    faith_succeeded = True
                    if not faithfulness.is_successful():
                        reasoning = faithfulness.reason or str(faithfulness.score)
                        red_cards.append(
                            f"Faithfulness Failure (DeepEval): {reasoning}"
                        )
                        if failure_reason == FailureReason.NONE:
                            failure_reason = FailureReason.FAITHFULNESS
                    break
                except Exception as eval_err:
                    if attempt < DEEPEVAL_MAX_RETRIES - 1:
                        logger.warning(
                            "DeepEval faithfulness attempt %d failed for %s: %s — retrying",
                            attempt + 1,
                            decision_id,
                            eval_err,
                        )
                        await asyncio.sleep(2)
                    else:
                        logger.error(
                            "DeepEval faithfulness failed for %s after %d attempts: %s",
                            decision_id,
                            DEEPEVAL_MAX_RETRIES,
                            eval_err,
                        )
                        red_cards.append(
                            f"DeepEval Faithfulness Error: {type(eval_err).__name__}: {eval_err}"
                        )
                        if failure_reason == FailureReason.NONE:
                            failure_reason = FailureReason.DEEPEVAL_ERROR

            # ── Answer Relevancy check (with retry + semaphore) ──
            for attempt in range(DEEPEVAL_MAX_RETRIES):
                try:
                    async with _deepeval_semaphore:
                        await asyncio.wait_for(
                            relevancy.a_measure(test_case),
                            timeout=DEEPEVAL_TIMEOUT_SEC,
                        )
                    logger.debug(
                        "DeepEval relevancy for %s: score=%.3f reason=%s",
                        decision_id,
                        relevancy.score or 0,
                        (relevancy.reason or "")[:200],
                    )
                    if not relevancy.is_successful():
                        reasoning = relevancy.reason or str(relevancy.score)
                        red_cards.append(
                            f"Answer Relevancy Failure (DeepEval): {reasoning}"
                        )
                        if failure_reason == FailureReason.NONE:
                            failure_reason = FailureReason.RELEVANCY
                    break
                except Exception as eval_err:
                    if attempt < DEEPEVAL_MAX_RETRIES - 1:
                        logger.warning(
                            "DeepEval relevancy attempt %d failed for %s: %s — retrying",
                            attempt + 1,
                            decision_id,
                            eval_err,
                        )
                        await asyncio.sleep(2)
                    else:
                        logger.error(
                            "DeepEval relevancy failed for %s after %d attempts: %s",
                            decision_id,
                            DEEPEVAL_MAX_RETRIES,
                            eval_err,
                        )
                        red_cards.append(
                            f"DeepEval Relevancy Error: {type(eval_err).__name__}: {eval_err}"
                        )
                        if failure_reason == FailureReason.NONE:
                            failure_reason = FailureReason.DEEPEVAL_ERROR

            # 6. ROUGE-L Grounding Check (Semantic/Text Overlap)
            # Extract meaningful reasoning text and use full context for fair comparison
            try:
                from rouge_score import rouge_scorer

                rouge_scorer_instance = rouge_scorer.RougeScorer(
                    ["rougeL"], use_stemmer=True
                )

                # Extract only the reasoning/rationale from the raw response
                # (strips code blocks, tool calls, JSON scaffolding)
                reasoning_text = extract_reasoning_text(raw_response or "")

                # Use the full context for ROUGE reference (not the 2000-char truncation)
                # Normalize both texts for fair comparison
                rouge_reference = full_context_blob or context_blob
                norm_prediction = normalize_for_rouge(reasoning_text)
                norm_reference = normalize_for_rouge(rouge_reference)

                if norm_prediction and norm_reference:
                    rouge_scores = rouge_scorer_instance.score(
                        norm_reference, norm_prediction
                    )
                    # Use PRECISION, not F-measure.  The bot's reasoning (prediction)
                    # is always much shorter than the full context blob (reference).
                    # F-measure's recall component measures "how much of the context
                    # is covered by reasoning" — this is always near zero because the
                    # context is 10-25 KB while reasoning is ~500-1000 chars.
                    # Precision answers the right question: "How much of what the bot
                    # said actually came from the context%s" — i.e., is it grounded%s
                    rouge_l = round(rouge_scores["rougeL"].precision, 3)
                else:
                    rouge_l = 0.0

                # Citation overlap: fraction of numbers in reasoning found in context
                citation_score = compute_citation_overlap(
                    reasoning_text, rouge_reference
                )

                # Composite grounding score: weighted blend of ROUGE-L and citation
                grounding_score = round(
                    ROUGE_WEIGHT * rouge_l + CITATION_WEIGHT * citation_score, 3
                )

                # Store all three metrics as top-level keys so the strategy
                # auditor (and any downstream consumer) can read each
                # independently.  Legacy key "hf_rougeL" kept for backward
                # compatibility with existing DB rows.
                oracle_results["checklist"]["hf_rougeL"] = grounding_score
                oracle_results["checklist"]["raw_rougeL"] = rouge_l
                oracle_results["checklist"]["citation_overlap"] = citation_score
                oracle_results["checklist"]["grounding_score"] = grounding_score
            except Exception as hf_err:
                logger.error(
                    f"ROUGE-L grounding check failed for {decision_id}: {hf_err}"
                )

            # Optional: You can enforce Policy rules here strictly.
            policy_understanding = True

            # 7. Prompt VLLM Model (with retry consistent with DeepEval pattern)
            eval_response = None
            for attempt in range(DEEPEVAL_MAX_RETRIES):
                try:
                    eval_response, tokens, ms = await asyncio.wait_for(
                        llm.chat(
                            system=SYSTEM_PROMPT,
                            user=user_prompt,
                            temperature=0.1,
                            max_tokens=256,
                            priority=Priority.HIGH,
                            agent_name="judge_evaluator",
                            ticker=ticker,
                        ),
                        timeout=DEEPEVAL_TIMEOUT_SEC,
                    )
                    break
                except Exception as api_err:
                    if attempt < DEEPEVAL_MAX_RETRIES - 1:
                        logger.warning(
                            "llm.chat attempt %d failed for %s: %s — retrying",
                            attempt + 1,
                            decision_id,
                            api_err,
                        )
                        await asyncio.sleep(2)
                    else:
                        logger.error(
                            "llm.chat failed for %s after %d attempts: %s",
                            decision_id,
                            DEEPEVAL_MAX_RETRIES,
                            api_err,
                        )
                        raise api_err

            payload = parse_json_response(eval_response)

            # 8. Calculate Final Hybrid Auto-Score
            # Hybrid formula: 50% deterministic data completeness, 50% subjective causation thesis
            llm_score = float(payload.get("judge_score", 0))
            oracle_score = float(oracle_results["completeness_score"])

            base_score = round((llm_score + oracle_score) / 2.0, 2)
            # Only zero on faithfulness red cards — not on data/parse issues
            # which are upstream problems, not hallucination.
            final_quality_score = 0 if red_cards else base_score

            # ── Build evidence_gathering with failure metadata ──
            evidence = oracle_results["checklist"].copy()
            if failure_reason != FailureReason.NONE:
                evidence["failure_reason"] = failure_reason.value

            evidence_json = json.dumps(evidence)
            red_cards_json = json.dumps(red_cards)

            # Upsert Evaluator Score
            existing = db.execute(
                "SELECT decision_id FROM decision_evaluations WHERE decision_id = %s",
                [decision_id],
            ).fetchone()

            if existing:
                db.execute(
                    """
                    UPDATE decision_evaluations SET
                        judge_a_score = %s,
                        final_quality_score = %s,
                        red_cards = %s,
                        first_principles_reasoning = %s,
                        policy_understanding = %s,
                        evidence_gathering = %s
                    WHERE decision_id = %s
                    """,
                    [
                        base_score,
                        final_quality_score,
                        red_cards_json,
                        payload.get("first_principles", ""),
                        policy_understanding,
                        evidence_json,
                        decision_id,
                    ],
                )
            else:
                db.execute(
                    """
                    INSERT INTO decision_evaluations (
                        decision_id, cycle_id, ticker, timestamp, 
                        judge_a_score, final_quality_score, red_cards, 
                        first_principles_reasoning, policy_understanding, discrepancy_trigger, evidence_gathering
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        decision_id,
                        cycle_id,
                        ticker,
                        created_at,
                        base_score,
                        final_quality_score,
                        red_cards_json,
                        payload.get("first_principles", ""),
                        policy_understanding,
                        False,
                        evidence_json,
                    ],
                )

            logger.info(
                f"Decision {decision_id} Auto-Evaluated: Score {final_quality_score}"
                f" | failure_reason={failure_reason.value}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed LLM-as-a-Judge for {decision_id}: {e}", exc_info=True)
            return False
