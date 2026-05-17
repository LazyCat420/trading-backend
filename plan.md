# Plan: Trading Run Fixes & Data Pipeline Consolidation

<architecture_plan>
1. **Remove Redundant Janitor Agent**: The `janitor_agent.py` and its invocation in `data_phase.py` will be completely removed. The summarizer will handle all data gating and discarding logic.
2. **Standardize Database Lookup**: We will strictly use the `ticker_metadata` table (or whichever table has a clean `name` column) for mapping ticker symbols to full company names. This lookup will be wrapped in a pure wrapper function.
3. **Fix the Infinite Summarizer/Offline Sync Loop**: The `_continuous_summarization` loop in `data_phase.py` hangs when the summarizer repeatedly fetches the same items due to a failed update, or it runs indefinitely if collection hangs. We will:
    - Add a `max_retries` counter or mark items as `failed` in the DB if the update fails, preventing the exact same articles from being reprocessed forever.
    - Suppress redundant `[PRISM] Offline log saved` prints if they are flooding the console, so the main pipeline progress is visible.
4. **No-Truncation Discard Visibility (Frontend)**: When data is discarded by the gatekeeper, it must be properly logged and surfaced to the user via the frontend (or console logs). CRITICAL: We will ensure that NO TRUNCATION occurs in the system prompts, user texts, or model reports when logging. The full story behind the discard must be preserved so the user can accurately evaluate the gatekeeper's decision.
5. **Update Summarizer System Prompt**: The prompt (`NEWS_SYSTEM_JSON` and `REDDIT_SYSTEM_JSON` in `summarizer.py`) will be relaxed. It will explicitly state: "If the data lacks technical details but still tells a meaningful story or narrative about the company's fundamentals, management, or market environment, you MUST ACCEPT IT and summarize the story."
</architecture_plan>

<data_and_interfaces>
```python
# In app/utils/db_metadata.py (New or updated wrapper file)
def get_company_name_from_ticker(ticker: str) -> str:
    """
    Queries the ticker_metadata table to return the full company name for a given ticker.
    Returns the ticker itself if no name is found.
    """
    pass

# In app/processors/summarizer.py
def _parse_reddit_json_response(response: str) -> dict:
    """
    Parses JSON. We will update this to ensure `discard` reasons are cleanly extracted
    and we will log a clear WARNING/INFO indicating what was discarded and why.
    """
    pass

def log_discarded_item(item_id: str, reason: str, full_original_text: str) -> None:
    """
    Logs the discarded item to a visible channel (frontend evaluation log or db)
    so the user can audit the gatekeeper's decisions. 
    Crucially: the full_original_text is NOT truncated.
    """
    pass
```
</data_and_interfaces>

<implementation>
**Step 1**: Delete `app/pipeline/data/data_janitor.py` and remove references to it in `data_phase.py` (Pass 1.6).

**Step 2**: In `app/processors/summarizer.py`, update `NEWS_SYSTEM_JSON`, `REDDIT_SYSTEM_JSON`, and `REDDIT_CONSOLIDATED_SYSTEM`:
- Add instructions: "Do NOT discard if the text is vague but contains a relevant narrative/story that impacts fundamentals. Summarize the story."
- Include dynamic context via the `get_company_name_from_ticker` function, so it knows "NVO" is "Novo Nordisk".

**Step 3**: In `summarizer.py`, inside `_summarize_news_batch` and `_summarize_reddit_batch`, whenever `action == "discard"`, invoke `log_discarded_item` to output a highly visible report (and store it for frontend evaluation) that includes the exact reason and the full, UNTRUNCATED original text. Ensure no legacy 150-word truncations exist in this pipeline.

**Step 4**: In `data_phase.py`, modify the `_continuous_summarization` `while` loop to ensure it has a maximum iteration count or gracefully tracks failure states so it does not endlessly query and fail on the same rows, which creates the "offline sync" log flood.

**Step 5**: Wait for the USER to audit the database logs to identify specific edge cases for context enrichment failures before deploying specific alias rules.
</implementation>
