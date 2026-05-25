import pytest
import os
import shutil
from app.tools.wiki_tools import write_memory_note, read_memory_note, WIKI_DIR

@pytest.mark.asyncio
async def test_write_memory_note_robustness():
    # Ensure clean test environment directory
    test_wiki_dir = os.path.join(os.getcwd(), "memory", "LLMWiki")
    if os.path.exists(test_wiki_dir):
        shutil.rmtree(test_wiki_dir)

    # 1. Standard call
    res = await write_memory_note(topic="AAPL_strategy", content="Buy AAPL at support")
    assert "Success" in res
    assert os.path.exists(os.path.join(test_wiki_dir, "AAPL_strategy.md"))
    
    # Check read
    read_res = await read_memory_note(topic="AAPL_strategy")
    assert "Buy AAPL at support" in read_res

    # 2. Call with only content and topic in kwargs as "title"
    res = await write_memory_note(content="Sell strategy", title="NVDA_strategy")
    assert "Success" in res
    assert os.path.exists(os.path.join(test_wiki_dir, "NVDA_strategy.md"))
    
    read_res = await read_memory_note(title="NVDA_strategy")
    assert "Sell strategy" in read_res

    # 3. Call with content as "note" and topic in kwargs as "ticker"
    res = await write_memory_note(note="Focus on RSI", ticker="MSFT")
    assert "Success" in res
    assert os.path.exists(os.path.join(test_wiki_dir, "MSFT.md"))
    
    read_res = await read_memory_note(ticker="MSFT")
    assert "Focus on RSI" in read_res

    # 4. Call missing content
    res = await write_memory_note(topic="MissingContent")
    assert "Error" in res

    # 5. Call missing topic but has content (should default to general_note)
    res = await write_memory_note(note="General test note content")
    assert "Success" in res
    assert os.path.exists(os.path.join(test_wiki_dir, "general_note.md"))

    # Cleanup after test
    if os.path.exists(test_wiki_dir):
        shutil.rmtree(test_wiki_dir)
