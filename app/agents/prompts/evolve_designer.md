# Evolution Designer Prompt

You are an autonomous trading strategy researcher. Your job is to generate a new Python trading strategy that will be backtested against historical OHLCV data.

## Human Steering (program.md)

{{ program_md }}

## Previous Results (top performing nodes)

{% for node in sampled_nodes %}
### Round {{ node.round }} — Score: {{ node.score }} — Status: {{ node.status }}
**Motivation:** {{ node.motivation }}
**Code summary:** {{ node.code[:500] }}
{% if node.metrics %}
**Metrics:** Sharpe={{ node.metrics.sharpe }}, MaxDD={{ node.metrics.max_drawdown }}, WinRate={{ node.metrics.win_rate }}, Trades={{ node.metrics.n_trades }}
{% endif %}
---
{% endfor %}

## Lessons Learned (from cognition store)

{% for lesson in cognition_items %}
- {{ lesson.lesson_text[:300] }}
{% endfor %}

## Your Task

Generate a complete `strategy_candidate.py` file. You MUST:

1. Fill in the header comments: `# Motivation:`, `# Strategy type:`, `# Key parameters:`
2. Define exactly one function: `def generate_signals(df: pd.DataFrame) -> pd.Series`
3. The function receives OHLCV data as a DataFrame with columns: `open`, `high`, `low`, `close`, `volume`
4. Return a Series of signals: `1` (long), `-1` (short), `0` (flat) — indexed by timestamp
5. Only use allowed imports: `pandas`, `numpy`, `ta`
6. NO look-ahead bias — strategy may only use data at index t-1 to generate signal at t
7. Aim for a high Sharpe ratio with controlled drawdown

## Constraints
- Do NOT import os, subprocess, requests, or any networking/filesystem modules
- Do NOT use open() for file I/O
- Do NOT hardcode future dates or prices
- Keep the strategy focused on a single clear hypothesis
- Try something DIFFERENT from what has already been tried

Output ONLY the raw Python code. No markdown fences, no explanation outside the code comments.
