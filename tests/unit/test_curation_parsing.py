"""
Test: Curation Pass JSON Parsing Hardening.

Verifies that _parse_curation_response correctly handles various
LLM output formats including think tags, markdown, and commentary.
"""



def _parse(content, valid=None):
    """Helper to call _parse_curation_response."""
    from app.pipeline.analysis.curation_pass import _parse_curation_response
    if valid is None:
        valid = ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA"]
    return _parse_curation_response(content, valid)


class TestCurationParsing:

    def test_clean_json(self):
        """Standard clean JSON response."""
        content = '{"promote": ["AAPL", "MSFT"], "skip": ["GOOG"], "reasoning": {}}'
        result = _parse(content)
        assert result == ["AAPL", "MSFT"]

    def test_markdown_wrapped_json(self):
        """JSON wrapped in markdown code block."""
        content = '```json\n{"promote": ["TSLA"], "skip": [], "reasoning": {}}\n```'
        result = _parse(content)
        assert result == ["TSLA"]

    def test_think_tags_with_json(self):
        """Qwen3-style response with <think> block before JSON."""
        content = (
            "<think>Let me analyze the tickers...</think>\n"
            '{"promote": ["NVDA"], "skip": ["AAPL"], '
            '"reasoning": {"NVDA": "Strong momentum"}}'
        )
        result = _parse(content)
        assert result == ["NVDA"]

    def test_commentary_before_json(self):
        """LLM adds commentary before the JSON."""
        content = (
            "Based on my analysis, here are my recommendations:\n\n"
            '{"promote": ["AAPL", "GOOG"], "skip": ["TSLA"], "reasoning": {}}'
        )
        result = _parse(content)
        assert result == ["AAPL", "GOOG"]

    def test_commentary_after_json(self):
        """LLM adds commentary after the JSON."""
        content = (
            '{"promote": ["MSFT"], "skip": [], "reasoning": {}}\n\n'
            "I hope this helps with your trading decisions."
        )
        result = _parse(content)
        assert result == ["MSFT"]

    def test_invalid_ticker_filtered(self):
        """Tickers not in the valid list should be filtered out."""
        content = '{"promote": ["AAPL", "FAKE", "XYZ123"], "skip": [], "reasoning": {}}'
        result = _parse(content)
        assert result == ["AAPL"]

    def test_empty_promote_list(self):
        """Empty promote list is valid."""
        content = '{"promote": [], "skip": ["AAPL"], "reasoning": {}}'
        result = _parse(content)
        assert result == []

    def test_completely_invalid_response(self):
        """Completely non-JSON response returns empty list."""
        content = "I think AAPL is a great buy. MSFT also looks promising."
        result = _parse(content)
        assert result == []

    def test_promote_not_a_list(self):
        """If 'promote' is not a list, return empty."""
        content = '{"promote": "AAPL", "skip": [], "reasoning": {}}'
        result = _parse(content)
        assert result == []

    def test_case_normalization(self):
        """Tickers should be uppercased."""
        content = '{"promote": ["aapl", "msft"], "skip": [], "reasoning": {}}'
        result = _parse(content)
        assert result == ["AAPL", "MSFT"]

    def test_nested_json_in_think_block(self):
        """JSON inside unclosed think block should still be extracted."""
        content = (
            "<think>The analysis suggests:\n"
            '{"action": "review"}\n'
            "</think>\n"
            '{"promote": ["GOOG"], "skip": [], "reasoning": {"GOOG": "AI leader"}}'
        )
        result = _parse(content)
        assert result == ["GOOG"]
