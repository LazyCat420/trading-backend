"""
Tests for the Reddit summarizer rebuild.

Tests cover:
  Phase 1: Post density classification, pre-filtering, consolidation
  Phase 2: JSON quality gating, sentiment extraction, output validation
  Phase 3: Context builder integration (uses summary, skips discarded)

Run:
    pytest tests/unit/test_summarizer_reddit.py -v
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# ── Phase 1 Tests: Classification & Pre-Filtering ──────────────────────


class TestPostDensityClassification:
    """Tests for _classify_reddit_density — the pre-filter gate."""

    def test_classify_substantial_long_body(self):
        """Post with 300+ chars body → 'substantial'."""
        from app.processors.summarizer import _classify_reddit_density

        body = "A" * 250  # well above MIN_SUBSTANTIAL_BODY_CHARS (200)
        assert _classify_reddit_density("Title", body) == "substantial"

    def test_classify_substantial_title_plus_body(self):
        """Post where title + body >= 300 chars → 'substantial'."""
        from app.processors.summarizer import _classify_reddit_density

        title = "A" * 150
        body = "B" * 160  # total = 310 >= MIN_SUBSTANTIAL_TOTAL_CHARS (300)
        assert _classify_reddit_density(title, body) == "substantial"

    def test_classify_thin_medium_body(self):
        """Post with 80 chars body (< 200) and short title → 'thin'."""
        from app.processors.summarizer import _classify_reddit_density

        body = "GOOGL is looking strong after earnings but I'm still cautious about their cloud revenue growth"
        assert len(body) < 200
        assert _classify_reddit_density("Short title", body) == "thin"

    def test_classify_garbage_very_short(self):
        """Post with < 30 chars body AND < 20 chars title → 'garbage'."""
        from app.processors.summarizer import _classify_reddit_density

        assert _classify_reddit_density("Short", "Tiny body") == "garbage"

    def test_classify_garbage_removed(self):
        """[removed] body → 'garbage'."""
        from app.processors.summarizer import _classify_reddit_density

        assert _classify_reddit_density("GOOGL DD", "[removed]") == "garbage"

    def test_classify_garbage_deleted(self):
        """[deleted] body → 'garbage'."""
        from app.processors.summarizer import _classify_reddit_density

        assert _classify_reddit_density("GOOGL DD", "[deleted]") == "garbage"

    def test_classify_garbage_empty_body(self):
        """Empty body → 'garbage'."""
        from app.processors.summarizer import _classify_reddit_density

        assert _classify_reddit_density("GOOGL DD", "") == "garbage"

    def test_classify_garbage_scrape_artifact(self):
        """Cloudflare challenge page → 'garbage'."""
        from app.processors.summarizer import _classify_reddit_density

        body = "Checking your browser before accessing the site. Please enable javascript."
        assert _classify_reddit_density("Title", body) == "garbage"

    def test_classify_thin_has_enough_title(self):
        """Short body but decent title (>= 20 chars) → 'thin', not 'garbage'."""
        from app.processors.summarizer import _classify_reddit_density

        # Body is 35 chars (above MIN_THIN_BODY_CHARS=30), title is long
        body = "Was rocky but we sooo back finally"
        title = "I believe in GOOGL long term growth potential"
        result = _classify_reddit_density(title, body)
        assert result == "thin"  # Not garbage — has enough substance to try

    def test_classify_boundary_exactly_200_body(self):
        """Body exactly 200 chars → 'substantial'."""
        from app.processors.summarizer import _classify_reddit_density

        body = "A" * 200
        assert _classify_reddit_density("Title", body) == "substantial"

    def test_classify_boundary_just_under_200_body(self):
        """Body 199 chars with short title → 'thin'."""
        from app.processors.summarizer import _classify_reddit_density

        body = "A" * 199
        assert _classify_reddit_density("T", body) == "thin"


# ── Phase 2 Tests: JSON Output Parsing ─────────────────────────────────


class TestRedditJsonParsing:
    """Tests for _parse_reddit_json_response."""

    def test_parse_valid_accept(self):
        """Standard accept response with all fields."""
        from app.processors.summarizer import _parse_reddit_json_response

        response = json.dumps({
            "action": "accept",
            "reason": "Strong thesis with data",
            "confidence": 85,
            "summary": "- GOOGL making custom CPUs (Axion) to reduce Intel dependence\n- Bullish on vertically integrated AI hardware play",
            "sentiment": "bullish",
            "tickers": ["GOOGL"],
        })
        result = _parse_reddit_json_response(response)
        assert result["q_status"] == "accepted"
        assert result["q_score"] == 85
        assert "GOOGL" in result["tickers_mentioned"]
        assert result["sentiment"] == "bullish"
        assert len(result["summary"]) > 30

    def test_parse_valid_discard(self):
        """Standard discard response."""
        from app.processors.summarizer import _parse_reddit_json_response

        response = json.dumps({
            "action": "discard",
            "reason": "Pure meme hype, no thesis",
            "confidence": 90,
            "summary": "",
            "sentiment": "neutral",
            "tickers": [],
        })
        result = _parse_reddit_json_response(response)
        assert result["q_status"] == "discarded"
        assert result["summary"] == ""

    def test_parse_markdown_fenced_json(self):
        """JSON inside ```json ... ``` fences."""
        from app.processors.summarizer import _parse_reddit_json_response

        response = '```json\n{"action":"accept","reason":"Good DD","confidence":75,"summary":"- Deep analysis of NVDA earnings beat\\n- Price target $150 based on forward PE","sentiment":"bullish","tickers":["NVDA"]}\n```'
        result = _parse_reddit_json_response(response)
        assert result["q_status"] == "accepted"
        assert "NVDA" in result["tickers_mentioned"]

    def test_parse_garbage_response(self):
        """Non-JSON LLM response → discarded."""
        from app.processors.summarizer import _parse_reddit_json_response

        response = "Let me create that agent for you. Looking at this post..."
        result = _parse_reddit_json_response(response)
        assert result["q_status"] == "discarded"
        assert "parse failed" in result["q_reason"]

    def test_parse_short_summary_discarded(self):
        """Accept with < 30 char summary → gets reclassified as discarded."""
        from app.processors.summarizer import _parse_reddit_json_response

        response = json.dumps({
            "action": "accept",
            "reason": "OK",
            "confidence": 60,
            "summary": "Good job!",  # way too short
            "sentiment": "neutral",
            "tickers": [],
        })
        result = _parse_reddit_json_response(response)
        assert result["q_status"] == "discarded"
        assert "too short" in result["q_reason"]

    def test_parse_summary_as_list(self):
        """Summary field is a JSON array → joined with bullet points."""
        from app.processors.summarizer import _parse_reddit_json_response

        response = json.dumps({
            "action": "accept",
            "reason": "Multi-point analysis",
            "confidence": 80,
            "summary": [
                "GOOGL building custom Axion CPUs to compete with Intel",
                "Bullish thesis: vertical integration reduces cloud costs",
            ],
            "sentiment": "bullish",
            "tickers": ["GOOGL"],
        })
        result = _parse_reddit_json_response(response)
        assert result["q_status"] == "accepted"
        assert "Axion" in result["summary"]
        assert result["summary"].count("-") >= 2

    def test_parse_missing_action_field(self):
        """JSON without 'action' field → discarded."""
        from app.processors.summarizer import _parse_reddit_json_response

        response = json.dumps({"summary": "something", "confidence": 50})
        result = _parse_reddit_json_response(response)
        assert result["q_status"] == "discarded"


# ── Phase 2 Tests: Sentiment Mapping ─────────────────────────────────


class TestSentimentMapping:
    """Tests for sentiment label → numeric score conversion."""

    def test_sentiment_bullish(self):
        from app.processors.summarizer import _SENTIMENT_SCORE_MAP

        assert _SENTIMENT_SCORE_MAP["bullish"] == 0.8

    def test_sentiment_bearish(self):
        from app.processors.summarizer import _SENTIMENT_SCORE_MAP

        assert _SENTIMENT_SCORE_MAP["bearish"] == 0.2

    def test_sentiment_neutral(self):
        from app.processors.summarizer import _SENTIMENT_SCORE_MAP

        assert _SENTIMENT_SCORE_MAP["neutral"] == 0.5

    def test_sentiment_mixed(self):
        from app.processors.summarizer import _SENTIMENT_SCORE_MAP

        assert _SENTIMENT_SCORE_MAP["mixed"] == 0.5


# ── Integration Tests: Reddit Batch Summarizer ──────────────────────


class TestRedditBatchSummarizer:
    """Tests for the rebuilt _summarize_reddit_batch with mocked LLM."""

    @pytest.mark.asyncio
    async def test_garbage_post_no_llm_call(self):
        """Garbage posts should be discarded without making any LLM call."""
        from app.processors.summarizer import _summarize_reddit_batch

        rows = [
            ("id1", "Title", "[removed]", "wallstreetbets"),
        ]

        with patch("app.processors.summarizer._summarize_one") as mock_llm:
            results = await _summarize_reddit_batch(rows)

        # No LLM call should have been made
        mock_llm.assert_not_called()
        assert len(results) == 1
        assert results[0]["q_status"] == "discarded"
        assert "removed" in results[0]["q_reason"]

    @pytest.mark.asyncio
    async def test_substantial_post_gets_json_summary(self):
        """Substantial post → individual LLM call with JSON quality gating."""
        from app.processors.summarizer import _summarize_reddit_batch

        long_body = (
            "Google literally makes its own CPUs (Axion), not just TPUs. "
            "Why is $GOOGL not mooning like Intel/AMD on 'CPU for AI' trend? "
            "Market finally figured out that CPUs are just as important as GPU "
            "and TPU for AI because that's where the actual code execution happens. "
            "The thing is Google literally built its own custom CPU called Axion "
            "specifically to stop paying the 'Intel tax' for its cloud. "
        )

        rows = [("id1", "GOOGL Custom CPU Play", long_body, "stocks")]

        mock_response = json.dumps({
            "action": "accept",
            "reason": "Strong thesis on GOOGL hardware vertical integration",
            "confidence": 82,
            "summary": "- GOOGL building custom Axion CPUs to compete with Intel/AMD in cloud computing\n- Bullish thesis: vertical integration reduces 'Intel tax' and positions GOOGL as most vertically integrated AI company",
            "sentiment": "bullish",
            "tickers": ["GOOGL"],
        })

        with patch("app.processors.summarizer._summarize_one", return_value=(mock_response, 150)):
            with patch("app.services.adaptive_concurrency.concurrency_controller") as mock_cc:
                # Make gather return the coroutine results directly
                async def fake_gather(tasks, label=""):
                    return [await t for t in tasks]
                mock_cc.gather = fake_gather

                results = await _summarize_reddit_batch(rows)

        assert len(results) == 1
        assert results[0]["q_status"] == "accepted"
        assert "GOOGL" in results[0].get("tickers_mentioned", "")

    @pytest.mark.asyncio
    async def test_thin_posts_consolidated(self):
        """Multiple thin posts from same subreddit → consolidated into one LLM call."""
        from app.processors.summarizer import _summarize_reddit_batch

        rows = [
            ("id1", "GOOGL to the moon", "Was rocky but we sooo back again now", "wallstreetbets"),
            ("id2", "I believe in GOOGL", "Best tech company for the long run", "wallstreetbets"),
            ("id3", "GOOGL earnings play", "Calls looking good for next quarter?", "wallstreetbets"),
        ]

        mock_response = json.dumps({
            "action": "accept",
            "reason": "Collective bullish sentiment on GOOGL",
            "confidence": 65,
            "summary": "- Collective bullish sentiment on GOOGL across multiple r/wallstreetbets posts\n- Sentiment focused on long-term growth thesis and upcoming earnings",
            "sentiment": "bullish",
            "tickers": ["GOOGL"],
        })

        call_count = 0

        async def mock_summarize(system, user_text, agent_name="", ticker="", cycle_id=""):
            nonlocal call_count
            call_count += 1
            return mock_response, 100

        with patch("app.processors.summarizer._summarize_one", side_effect=mock_summarize):
            with patch("app.services.adaptive_concurrency.concurrency_controller") as mock_cc:
                async def fake_gather(tasks, label=""):
                    return [await t for t in tasks]
                mock_cc.gather = fake_gather

                results = await _summarize_reddit_batch(rows)

        # All 3 thin posts should share the same consolidated summary
        assert len(results) == 3
        for r in results:
            assert r["q_status"] == "accepted"
            assert "consolidated" in r["q_reason"]

        # Only 1 LLM call should have been made (consolidated)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_mixed_density_batch(self):
        """Batch with garbage, thin, and substantial posts all handled correctly."""
        from app.processors.summarizer import _summarize_reddit_batch

        rows = [
            ("id_garbage", "Lol", "[deleted]", "wallstreetbets"),
            ("id_thin", "GOOGL bull", "Still bullish on this one for sure", "stocks"),
            ("id_substantial", "Deep GOOGL Analysis", "A" * 250, "stocks"),
        ]

        mock_response = json.dumps({
            "action": "accept",
            "reason": "Has thesis",
            "confidence": 70,
            "summary": "- Analysis of GOOGL with detailed coverage of fundamentals and growth trajectory",
            "sentiment": "bullish",
            "tickers": ["GOOGL"],
        })

        with patch("app.processors.summarizer._summarize_one", return_value=(mock_response, 100)):
            with patch("app.services.adaptive_concurrency.concurrency_controller") as mock_cc:
                async def fake_gather(tasks, label=""):
                    return [await t for t in tasks]
                mock_cc.gather = fake_gather

                results = await _summarize_reddit_batch(rows)

        # Should have results for all 3 posts
        assert len(results) == 3

        # Find each by ID
        by_id = {r["id"]: r for r in results}
        assert by_id["id_garbage"]["q_status"] == "discarded"
        assert by_id["id_substantial"]["q_status"] == "accepted"
        # Thin post gets individual treatment (only 1 in that subreddit group)
        assert by_id["id_thin"]["q_status"] == "accepted"


# ── Root Cause Fix Test: strip_think_tags ─────────────────────────────


class TestStripThinkTags:
    """Verify that _summarize_one strips <think> blocks."""

    @pytest.mark.asyncio
    async def test_think_tags_stripped(self):
        """LLM response with <think>...</think> should have thinking block removed."""
        from app.processors.summarizer import _summarize_one

        think_response = (
            "<think>Let me analyze this Reddit post about GOOGL. The user seems "
            "bullish but doesn't provide much data. I should summarize...</think>"
            '{"action":"accept","summary":"- Bullish on GOOGL vertical integration","confidence":70,"tickers":["GOOGL"],"sentiment":"bullish","reason":"Has thesis"}'
        )

        with patch("app.services.vllm_client.llm") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=(think_response, 200, 1500))
            text, tokens = await _summarize_one(
                "system prompt", "user text", "test_agent"
            )

        # The <think> block should be stripped — only JSON remains
        assert "<think>" not in text
        assert "</think>" not in text
        assert '{"action"' in text

    @pytest.mark.asyncio
    async def test_no_think_tags_passthrough(self):
        """Response without <think> tags passes through unchanged."""
        from app.processors.summarizer import _summarize_one

        clean_response = '{"action":"accept","summary":"- GOOGL is strong","confidence":80}'

        with patch("app.services.vllm_client.llm") as mock_llm:
            mock_llm.chat = AsyncMock(return_value=(clean_response, 100, 500))
            text, tokens = await _summarize_one(
                "system prompt", "user text", "test_agent"
            )

        assert text == clean_response
