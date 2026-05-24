"""
Data Summarizer — LLM-powered summaries for scraped content.

Runs as Pass 5.5 in the data pipeline. Finds items without summaries,
dispatches LLM calls in parallel (8 concurrent slots), stores results in DB.

Benefits:
  - 97% token reduction for YouTube (15K chars -> 500 chars)
  - 85% reduction for Reddit
  - Summaries reused by chat, context_builder, decision_engine
  - Non-English content auto-translated to English

Usage:
    from app.processors.summarizer import summarize_unsummarized
    stats = await summarize_unsummarized(emit)
"""

import asyncio
import json
import logging
import re
import time
from typing import Callable

from app.db.connection import get_db
from app.utils.text_utils import strip_think_tags, is_scrape_artifact

logger = logging.getLogger(__name__)


# ── Prompt Templates ──────────────────────────────────────────────

YOUTUBE_SYSTEM = """CRITICAL CONTEXT: You are an autonomous data processing script working for a quantitative trading firm. You are NOT a conversational chatbot. Do NOT talk to the user, give advice, or answer questions. Your ONLY purpose is to extract structured financial data to make profitable trading decisions.

Summarize the YouTube transcript below into 3-5 concise bullet points.

FOCUS ON:
- Stock tickers mentioned and their context (bullish/bearish/neutral)
- Specific price targets, earnings numbers, or financial metrics
- Key catalysts, events, or news mentioned
- Overall market sentiment and thesis

If the transcript is in a non-English language, translate it to English.
List any stock tickers mentioned at the end in format: TICKERS: AAPL, NVDA, TSLA

Keep your summary under 400 words. Be specific — cite numbers when available."""

REDDIT_SYSTEM = """CRITICAL CONTEXT: You are an autonomous data processing script working for a quantitative trading firm. You are NOT a conversational chatbot. Do NOT talk to the user, give advice, or answer questions. Your ONLY purpose is to extract structured financial data to make profitable trading decisions.

Summarize this Reddit post into 2-4 concise bullet points.

FOCUS ON:
- The main thesis or claim
- Stock tickers discussed and sentiment (bullish/bearish)
- Any specific data points, catalysts, or price targets
- Key discussion themes

Keep your summary under 200 words. Be specific."""

REDDIT_SYSTEM_JSON = """CRITICAL CONTEXT: You are an autonomous data processing script working for a quantitative trading firm. You are NOT a conversational chatbot. Do NOT converse, answer rhetorical questions from the post, or offer advice. Your ONLY purpose is to extract structured financial data to make profitable trading decisions.

You are a financial data quality gatekeeper and summarizer.
Evaluate the following Reddit post(s). Discard if it lacks substance
(pure meme with no thesis, emoji-only hype, no identifiable financial claim,
or too vague to extract actionable information).
Accepted posts MUST have a clear financial thesis, specific company/ticker reference,
or concrete data points.

Return ONLY a valid JSON object matching this schema:
{
  "action": "accept" or "discard",
  "reason": "Brief reason for your decision",
  "confidence": <integer 1-100>,
  "summary": "2-3 concise bullet points focusing on thesis, sentiment, and data. Leave empty if discarded.",
  "sentiment": "bullish" or "bearish" or "neutral" or "mixed",
  "tickers": ["GOOGL", "NVDA"]
}
Ensure the output is strictly valid JSON without any markdown formatting around it."""

REDDIT_CONSOLIDATED_SYSTEM = """CRITICAL CONTEXT: You are an autonomous data processing script working for a quantitative trading firm. You are NOT a conversational chatbot. Do NOT converse, answer questions, or offer advice. Your ONLY purpose is to extract structured financial data to make profitable trading decisions.

You are a financial data analyst. You are given multiple short Reddit posts
about the same stock. Individually each post lacks depth, but together they form
a sentiment picture. Synthesize them into a single consolidated summary.

Return ONLY a valid JSON object:
{
  "action": "accept" or "discard",
  "reason": "Brief reason",
  "confidence": <integer 1-100>,
  "summary": "2-4 bullet points synthesizing the collective sentiment, key claims, and any data points across all posts.",
  "sentiment": "bullish" or "bearish" or "neutral" or "mixed",
  "tickers": ["GOOGL"]
}
If the posts collectively still lack any financial substance, discard them."""

# ── Post Density Classification ──────────────────────────────────
# Controls whether a Reddit post is summarized individually,
# consolidated with similar thin posts, or discarded outright.
MIN_SUBSTANTIAL_BODY_CHARS = 200    # body alone >= this → individual summary
MIN_SUBSTANTIAL_TOTAL_CHARS = 300   # title + body >= this → individual summary
MIN_THIN_BODY_CHARS = 30            # body < this → garbage
MAX_CONSOLIDATION_BATCH = 10        # max thin posts per consolidation call

_SENTIMENT_SCORE_MAP = {
    "bullish": 0.8,
    "bearish": 0.2,
    "neutral": 0.5,
    "mixed": 0.5,
}

NEWS_SYSTEM_JSON = """CRITICAL CONTEXT: You are an autonomous data processing script working for a quantitative trading firm. You are NOT a conversational chatbot. Do NOT converse, answer questions, or offer advice. Your ONLY purpose is to extract structured financial data to make profitable trading decisions.

You are a financial data quality gatekeeper and summarizer.
Evaluate the following news article. Discard it if it lacks substance (e.g., is a teaser snippet, pure clickbait, mostly repeats the title, or is a placeholder/scrape artifact).
Accepted articles MUST have concrete facts, company references, or market conclusions.

Return ONLY a valid JSON object matching this schema:
{
  "action": "accept" or "discard",
  "reason": "Brief reason for your decision",
  "confidence": <integer 1-100>,
  "time_horizon": "short-term (e.g. daily swings/catalysts) OR long-term (e.g. YTD trends/structural shifts) OR both",
  "summary": "2-3 concise bullet points focusing on news events, impacts, and market implications. Explicitly note if the catalyst is short-term noise vs long-term fundamentals. Leave empty if discarded.",
  "tickers": ["AAPL", "NVDA"]
}
Ensure the output is strictly valid JSON without any markdown formatting around it, or use standard ```json blocks."""


# ── Core Summarization Functions ──────────────────────────────────


async def _summarize_one(
    system: str,
    user_text: str,
    agent_name: str,
    ticker: str = "",
    cycle_id: str = "",
) -> tuple[str, int]:
    """Call LLM for one summary. Returns (summary_text, tokens_used)."""
    from app.services.vllm_client import llm, Priority

    try:
        from app.services.prism_agent_caller import call_prism_agent
        response, tokens, elapsed_ms = await call_prism_agent(
            agent_id="CUSTOM_SUMMARIZER_AGENT",
            user_message=user_text,
            fallback_system_prompt=system,
            fallback_agent_name=agent_name,
            max_tokens=512,
            temperature=0.2,
            priority=Priority.LOW,
            ticker=ticker,
            cycle_id=cycle_id,
        )
        # Strip <think>...</think> blocks from Qwen3-style responses.
        # Without this, the entire chain-of-thought gets stored as the "summary"
        # and the actual content (after </think>) is lost or truncated.
        cleaned = strip_think_tags(response).strip()
        return cleaned, tokens
    except Exception as e:
        logger.warning("[summarizer] LLM call failed for %s: %s", agent_name, e)
        return "", 0


async def _summarize_youtube_batch(
    rows: list,
    cycle_id: str = "",
) -> list[dict]:
    """Summarize multiple YouTube transcripts in parallel."""

    async def _do_one(video_id, title, channel, transcript):
        user_text = (
            f"Title: {title}\nChannel: {channel}\nTranscript:\n{transcript[:15000]}"
        )
        summary, tokens = await _summarize_one(
            YOUTUBE_SYSTEM,
            user_text,
            agent_name="summarizer_youtube",
            cycle_id=cycle_id,
        )
        # Extract tickers from the summary (look for TICKERS: line)
        tickers_mentioned = ""
        for line in summary.split("\n"):
            if line.strip().upper().startswith("TICKERS:"):
                tickers_mentioned = line.split(":", 1)[1].strip()
                break
        return {
            "video_id": video_id,
            "summary": summary,
            "tickers_mentioned": tickers_mentioned,
            "tokens": tokens,
        }

    from app.services.adaptive_concurrency import concurrency_controller
    tasks = [_do_one(r[0], r[1], r[2], r[3]) for r in rows]
    return await concurrency_controller.gather(tasks, label="summarizer_youtube")


def _classify_reddit_density(title: str, body: str) -> str:
    """Classify a Reddit post into 'substantial', 'thin', or 'garbage'.

    Substantial: enough content to summarize individually.
    Thin: too short alone, but can be consolidated with similar posts.
    Garbage: not worth processing at all.
    """
    body_clean = body.strip() if body else ""
    title_clean = title.strip() if title else ""

    # Immediate garbage checks
    if body_clean in ("[removed]", "[deleted]", ""):
        return "garbage"
    if is_scrape_artifact(body_clean):
        return "garbage"
    if len(body_clean) < MIN_THIN_BODY_CHARS and len(title_clean) < 20:
        return "garbage"

    # Substantial: enough for individual summary
    total_chars = len(title_clean) + len(body_clean)
    if len(body_clean) >= MIN_SUBSTANTIAL_BODY_CHARS:
        return "substantial"
    if total_chars >= MIN_SUBSTANTIAL_TOTAL_CHARS:
        return "substantial"

    return "thin"


def _parse_reddit_json_response(response: str) -> dict:
    """Parse structured JSON from the Reddit summarizer LLM response.

    Handles markdown fences, bare JSON, and extraction failures.
    Returns a dict with action, summary, sentiment, tickers, etc.
    """
    q_status = "discarded"
    q_reason = "json parse failed"
    q_score = 0
    summary_text = ""
    tickers_str = ""
    sentiment = "neutral"

    try:
        # Try markdown-fenced JSON first
        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE
        )
        if match:
            payload = match.group(1)
        else:
            payload = response

        data = json.loads(payload)
        action = data.get("action", "").lower()
        q_reason = data.get("reason", "")[:200]
        q_score = int(data.get("confidence", 0))
        sentiment = data.get("sentiment", "neutral").lower()

        if action == "accept":
            q_status = "accepted"
            summary_data = data.get("summary", "")
            if isinstance(summary_data, list):
                summary_text = "\n".join(
                    f"- {str(s)}" if not str(s).strip().startswith("-") else str(s)
                    for s in summary_data
                )
            else:
                summary_text = str(summary_data)

            t_list = data.get("tickers", [])
            if isinstance(t_list, list):
                tickers_str = ", ".join(str(t) for t in t_list)
            else:
                tickers_str = str(t_list)

            # Final guard: if the parsed summary is too short, it's not useful
            if len(summary_text.strip()) < 30:
                q_status = "discarded"
                q_reason = "summary too short after extraction"
                summary_text = ""
        else:
            q_status = "discarded"
            if not q_reason:
                q_reason = "llm rejected"

    except Exception as e:
        q_status = "discarded"
        q_reason = f"llm parse failed: {str(e)[:50]}"

    return {
        "summary": summary_text,
        "q_status": q_status,
        "q_reason": q_reason,
        "q_score": q_score,
        "tickers_mentioned": tickers_str,
        "sentiment": sentiment,
    }


async def _summarize_reddit_batch(
    rows: list,
    cycle_id: str = "",
) -> list[dict]:
    """Summarize Reddit posts with pre-filtering, consolidation, and quality gating.

    Flow:
      1. Classify each post as substantial/thin/garbage
      2. Garbage → immediately discard (no LLM call)
      3. Substantial → individual LLM summary with JSON quality gating
      4. Thin → group by ticker, consolidate, then summarize as a batch
    """
    results = []
    substantial_tasks = []
    thin_by_ticker: dict[str, list[tuple]] = {}  # ticker → [(id, title, body, sub)]

    # ── Step 1: Classify all posts ──
    for row in rows:
        post_id, title, body, subreddit = row[0], row[1], row[2], row[3]
        density = _classify_reddit_density(title, body)

        if density == "garbage":
            # Immediate discard — no LLM call needed
            reason = "body too short or removed"
            if body.strip() in ("[removed]", "[deleted]"):
                reason = "post removed/deleted"
            elif is_scrape_artifact(body):
                reason = "scrape artifact detected"
            results.append({
                "id": post_id,
                "summary": "",
                "tokens": 0,
                "q_status": "discarded",
                "q_reason": f"prefilter: {reason}",
                "q_score": 0,
                "tickers_mentioned": "",
                "sentiment": "neutral",
            })
            logger.debug("[summarizer] Reddit %s discarded: %s", post_id, reason)

        elif density == "thin":
            # Group thin posts by a rough ticker key for consolidation.
            # We don't have the ticker in this function's row data, so
            # use subreddit as the grouping key (posts in the same sub
            # about similar topics cluster naturally).
            # NOTE: the DB query joins on ticker via the reddit_posts table,
            # so we could extend the query to include ticker for better grouping.
            key = subreddit.lower()
            thin_by_ticker.setdefault(key, []).append((post_id, title, body, subreddit))

        else:  # substantial
            substantial_tasks.append((post_id, title, body, subreddit))

    # ── Step 2: Summarize substantial posts individually ──
    async def _do_one_substantial(post_id, title, body, subreddit):
        user_text = f"Subreddit: r/{subreddit}\nTitle: {title}\nPost:\n{body[:15000]}"
        response, tokens = await _summarize_one(
            REDDIT_SYSTEM_JSON,
            user_text,
            agent_name="summarizer_reddit",
            cycle_id=cycle_id,
        )
        parsed = _parse_reddit_json_response(response)
        parsed["id"] = post_id
        parsed["tokens"] = tokens
        return parsed

    if substantial_tasks:
        from app.services.adaptive_concurrency import concurrency_controller
        sub_tasks = [
            _do_one_substantial(pid, t, b, s)
            for pid, t, b, s in substantial_tasks
        ]
        sub_results = await concurrency_controller.gather(
            sub_tasks, label="summarizer_reddit"
        )
        results.extend(sub_results)

    # ── Step 3: Consolidate and summarize thin posts ──
    async def _do_consolidated(group_posts):
        """Consolidate a group of thin posts into one LLM call."""
        lines = []
        for pid, title, body, sub in group_posts[:MAX_CONSOLIDATION_BATCH]:
            body_preview = body.strip()[:200] if body else ""
            lines.append(
                f"Subreddit: r/{sub} | Title: \"{title}\" | "
                f"Post: \"{body_preview}\""
            )
        user_text = "\n---\n".join(lines)
        user_text = f"{len(lines)} related Reddit posts:\n\n" + user_text

        response, tokens = await _summarize_one(
            REDDIT_CONSOLIDATED_SYSTEM,
            user_text,
            agent_name="summarizer_reddit",
            cycle_id=cycle_id,
        )
        parsed = _parse_reddit_json_response(response)

        # Apply the consolidated summary to ALL posts in the group
        consolidated_results = []
        per_post_tokens = tokens // max(len(group_posts), 1)
        for pid, title, body, sub in group_posts[:MAX_CONSOLIDATION_BATCH]:
            entry = dict(parsed)  # shallow copy
            entry["id"] = pid
            entry["tokens"] = per_post_tokens
            if parsed["q_status"] == "accepted":
                entry["q_reason"] = f"consolidated ({len(group_posts)} posts)"
            consolidated_results.append(entry)
        return consolidated_results

    if thin_by_ticker:
        from app.services.adaptive_concurrency import concurrency_controller
        consolidation_tasks = []
        for _key, group_posts in thin_by_ticker.items():
            if len(group_posts) == 1:
                # Single thin post — try individual summary anyway
                pid, t, b, s = group_posts[0]
                consolidation_tasks.append(
                    _do_one_substantial(pid, t, b, s)
                )
            else:
                consolidation_tasks.append(_do_consolidated(group_posts))

        cons_raw = await concurrency_controller.gather(
            consolidation_tasks, label="summarizer_reddit_consolidate"
        )
        for item in cons_raw:
            if isinstance(item, list):
                results.extend(item)  # consolidated group
            elif isinstance(item, dict):
                results.append(item)  # single thin post

    return results


async def _summarize_news_batch(
    rows: list,
    cycle_id: str = "",
) -> list[dict]:
    """Summarize multiple news articles in parallel."""
    import json
    import re
    from app.utils.text_utils import is_scrape_artifact as _is_scrape_artifact

    async def _do_one(article_id, title, summary_raw):
        summary_raw = summary_raw or ""
        body_clean = summary_raw.strip()
        title_clean = title.strip() if title else ""

        # Deterministic Pre-filters
        if _is_scrape_artifact(body_clean):
            return {
                "id": article_id,
                "summary": "",
                "tickers_mentioned": "",
                "tokens": 0,
                "q_status": "discarded",
                "q_reason": "prefilter: scrape artifact",
                "q_score": 0,
            }

        if len(body_clean) < 40:
            return {
                "id": article_id,
                "summary": "",
                "tickers_mentioned": "",
                "tokens": 0,
                "q_status": "discarded",
                "q_reason": "prefilter: body too short",
                "q_score": 0,
            }

        if (
            title_clean
            and body_clean.startswith(title_clean)
            and len(body_clean) < len(title_clean) + 50
        ):
            return {
                "id": article_id,
                "summary": "",
                "tickers_mentioned": "",
                "tokens": 0,
                "q_status": "discarded",
                "q_reason": "prefilter: body repeats title",
                "q_score": 0,
            }

        user_text = f"Title: {title}\nArticle:\n{summary_raw[:15000]}"
        response, tokens = await _summarize_one(
            NEWS_SYSTEM_JSON,
            user_text,
            agent_name="summarizer_news",
            cycle_id=cycle_id,
        )

        q_status = "discarded"
        q_reason = "json parse failed"
        q_score = 0
        summary_text = ""
        tickers_str = ""

        try:
            match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE
            )
            if match:
                payload = match.group(1)
            else:
                payload = response

            data = json.loads(payload)
            action = data.get("action", "").lower()
            q_reason = data.get("reason", "")[:200]
            q_score = int(data.get("confidence", 0))

            if action == "accept":
                q_status = "accepted"
                summary_data = data.get("summary", "")
                if isinstance(summary_data, list):
                    summary_text = "\n".join(
                        f"- {str(s)}" if not str(s).strip().startswith("-") else str(s)
                        for s in summary_data
                    )
                else:
                    summary_text = str(summary_data)

                t_list = data.get("tickers", [])
                if isinstance(t_list, list):
                    tickers_str = ", ".join(str(t) for t in t_list)
                else:
                    tickers_str = str(t_list)
            else:
                q_status = "discarded"
                if not q_reason:
                    q_reason = "llm rejected"

        except Exception as e:
            q_status = "discarded"
            q_reason = f"llm parse failed: {str(e)[:50]}"

        return {
            "id": article_id,
            "summary": summary_text,
            "tickers_mentioned": tickers_str,
            "tokens": tokens,
            "q_status": q_status,
            "q_reason": q_reason,
            "q_score": q_score,
        }

    from app.services.adaptive_concurrency import concurrency_controller
    tasks = [_do_one(r[0], r[1], r[2]) for r in rows]
    return await concurrency_controller.gather(tasks, label="summarizer_news")


# ── Main Entry Point ──────────────────────────────────────────────


async def summarize_unsummarized(
    emit: Callable | None = None,
    max_items: int = 30,
    cycle_id: str = "",
    ticker: str | None = None,
) -> dict:
    """Find items without summaries, batch-summarize via LLM.

    Runs all three data sources in parallel batches.
    With 8 concurrent LLM slots, this saturates the GPU.

    Returns: {youtube: N, reddit: N, news: N, tokens: N}
    """
    if emit is None:
        emit = lambda *a, **kw: None

    with get_db() as db:
        stats = {"youtube": 0, "reddit": 0, "news": 0, "tokens": 0, "errors": 0}
        t0 = time.monotonic()

        # ── Ensure columns exist (safe ALTER IF NOT EXISTS) ──
        from app.utils.db_migrations import ensure_summary_columns

        ensure_summary_columns(db)

        # ── Find unsummarized YouTube transcripts ──
        if ticker:
            ticker = ticker.upper()
            yt_rows = db.execute(
                """
                SELECT video_id, title, channel, raw_transcript
                FROM youtube_transcripts
                WHERE summary IS NULL
                  AND summarized_at IS NULL
                  AND raw_transcript IS NOT NULL
                  AND LENGTH(raw_transcript) > 50
                  AND ticker = %s
                ORDER BY published_at DESC
                LIMIT %s
            """,
                [ticker, max_items],
            ).fetchall()
        else:
            yt_rows = db.execute(
                """
                SELECT video_id, title, channel, raw_transcript
                FROM youtube_transcripts
                WHERE summary IS NULL
                  AND summarized_at IS NULL
                  AND raw_transcript IS NOT NULL
                  AND LENGTH(raw_transcript) > 50
                ORDER BY published_at DESC
                LIMIT %s
            """,
                [max_items],
            ).fetchall()

        # ── Find unsummarized Reddit posts ──
        # Low minimum (20 chars) because Python-side classification handles
        # garbage/thin/substantial routing. This lets us still discover and
        # properly discard posts rather than ignoring them forever.
        if ticker:
            reddit_rows = db.execute(
                """
                SELECT id, title, COALESCE(body, ''), subreddit
                FROM reddit_posts
                WHERE summary IS NULL
                  AND quality_status IS NULL
                  AND (LENGTH(title) + LENGTH(COALESCE(body, ''))) > 20
                  AND ticker = %s
                ORDER BY created_utc DESC
                LIMIT %s
            """,
                [ticker, max_items],
            ).fetchall()
        else:
            reddit_rows = db.execute(
                """
                SELECT id, title, COALESCE(body, ''), subreddit
                FROM reddit_posts
                WHERE summary IS NULL
                  AND quality_status IS NULL
                  AND (LENGTH(title) + LENGTH(COALESCE(body, ''))) > 20
                ORDER BY created_utc DESC
                LIMIT %s
            """,
                [max_items],
            ).fetchall()

        # ── Find unsummarized news (with bad/missing summaries) ──
        if ticker:
            news_rows = db.execute(
                """
                SELECT id, title, COALESCE(summary, '')
                FROM news_articles
                WHERE llm_summary IS NULL
                  AND quality_status IS NULL
                  AND title IS NOT NULL
                  AND LENGTH(title) > 10
                  AND ticker = %s
                ORDER BY published_at DESC
                LIMIT %s
            """,
                [ticker, max_items],
            ).fetchall()
        else:
            news_rows = db.execute(
                """
                SELECT id, title, COALESCE(summary, '')
                FROM news_articles
                WHERE llm_summary IS NULL
                  AND quality_status IS NULL
                  AND title IS NOT NULL
                  AND LENGTH(title) > 10
                ORDER BY published_at DESC
                LIMIT %s
            """,
                [max_items],
            ).fetchall()

        total = len(yt_rows) + len(reddit_rows) + len(news_rows)
        if total == 0:
            logger.debug(f"[summarizer] All data already summarized{f' for ticker {ticker}' if ticker else ''}")
            emit(
                "collecting",
                "summarize",
                f"All scraped data already has summaries{f' for ticker {ticker}' if ticker else ''}",
                status="ok",
                data=stats,
            )
            return stats

        logger.debug(
            "[summarizer] Found %d YouTube, %d Reddit, %d news to summarize%s",
            len(yt_rows),
            len(reddit_rows),
            len(news_rows),
            f" for ticker {ticker}" if ticker else "",
        )
        emit(
            "collecting",
            "summarize",
            f"Summarizing {total} items ({len(yt_rows)} YT, "
            f"{len(reddit_rows)} Reddit, {len(news_rows)} news){f' for ticker {ticker}' if ticker else ''}...",
            status="running",
        )

        # ── Run all batches in parallel (saturate GPU) ──
        async def _empty():
            return []

        yt_results, reddit_results, news_results = await asyncio.gather(
            _summarize_youtube_batch(yt_rows, cycle_id) if yt_rows else _empty(),
            _summarize_reddit_batch(reddit_rows, cycle_id) if reddit_rows else _empty(),
            _summarize_news_batch(news_rows, cycle_id) if news_rows else _empty(),
        )

        # ── Write YouTube summaries to DB ──
        for r in yt_results:
            if isinstance(r, Exception):
                logger.warning("[summarizer] YT batch item failed: %s", r)
                stats["errors"] += 1
                continue
            try:
                db.execute(
                    """
                    UPDATE youtube_transcripts
                    SET summary = %s, tickers_mentioned = %s, summarized_at = CURRENT_TIMESTAMP
                    WHERE video_id = %s
                """,
                    [r["summary"] or None, r["tickers_mentioned"], r["video_id"]],
                )
                # Fix #7: If primary ticker was NULL but LLM found tickers, assign the first one
                if r["tickers_mentioned"]:
                    first_ticker = r["tickers_mentioned"].split(",")[0].strip()
                    if first_ticker:
                        db.execute(
                            """
                            UPDATE youtube_transcripts
                            SET ticker = %s
                            WHERE video_id = %s AND ticker IS NULL
                        """,
                            [first_ticker, r["video_id"]],
                        )
                stats["youtube"] += 1
                stats["tokens"] += r["tokens"]
            except Exception as e:
                logger.warning(
                    "[summarizer] YT write failed for %s: %s", r["video_id"], e
                )
                stats["errors"] += 1

        # ── Write Reddit summaries to DB ──
        for r in reddit_results:
            if isinstance(r, Exception):
                logger.warning("[summarizer] Reddit batch item failed: %s", r)
                stats["errors"] += 1
                continue
            try:
                sentiment_score = _SENTIMENT_SCORE_MAP.get(
                    r.get("sentiment", "neutral"), 0.5
                )
                db.execute(
                    """
                    UPDATE reddit_posts
                    SET summary = %s,
                        quality_status = %s,
                        quality_reason = %s,
                        quality_score = %s,
                        sentiment_score = %s,
                        summarized_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """,
                    [
                        r["summary"] or None,
                        r.get("q_status", "discarded"),
                        r.get("q_reason", ""),
                        r.get("q_score", 0),
                        sentiment_score,
                        r["id"],
                    ],
                )
                if r.get("q_status") == "accepted" and r["summary"]:
                    stats["reddit"] += 1
                stats["tokens"] += r.get("tokens", 0)
            except Exception as e:
                logger.warning(
                    "[summarizer] Reddit write failed for %s: %s", r["id"], e
                )
                stats["errors"] += 1

        # ── Write news summaries to DB ──
        for r in news_results:
            if isinstance(r, Exception):
                logger.warning("[summarizer] News batch item failed: %s", r)
                stats["errors"] += 1
                continue
            try:
                db.execute(
                    """
                    UPDATE news_articles
                    SET llm_summary = %s, 
                        quality_status = %s,
                        quality_reason = %s,
                        quality_score = %s,
                        summarized_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """,
                    [
                        r["summary"] or None,
                        r["q_status"],
                        r["q_reason"],
                        r["q_score"],
                        r["id"],
                    ],
                )

                if r["q_status"] == "accepted" and r.get("tickers_mentioned"):
                    first_ticker = r["tickers_mentioned"].split(",")[0].strip()
                    if first_ticker:
                        db.execute(
                            """
                            UPDATE news_articles
                            SET ticker = %s
                            WHERE id = %s AND ticker IS NULL
                        """,
                            [first_ticker, r["id"]],
                        )
                stats["news"] += 1
                stats["tokens"] += r["tokens"]
            except Exception as e:
                logger.warning("[summarizer] News write failed for %s: %s", r["id"], e)
                stats["errors"] += 1

        ms = int((time.monotonic() - t0) * 1000)
        stats["ms"] = ms
        total_done = stats["youtube"] + stats["reddit"] + stats["news"]
        logger.info(
            "[summarizer] Done: %d summaries in %dms (%d tokens)",
            total_done,
            ms,
            stats["tokens"],
        )
        emit(
            "collecting",
            "summarize",
            f"Summarized {total_done} items: "
            f"{stats['youtube']} YT, {stats['reddit']} Reddit, {stats['news']} news "
            f"({stats['tokens']} tokens, {ms}ms)",
            status="ok",
            data=stats,
            elapsed_ms=ms,
        )

        return stats


# _ensure_summary_columns removed — use app.utils.db_migrations.ensure_summary_columns
