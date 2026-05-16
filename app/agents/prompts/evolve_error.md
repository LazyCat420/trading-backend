# Evolution Error Handler Prompt

You are a debugging expert. A generated trading strategy failed during execution. Diagnose the specific failure and suggest a targeted fix.

## Generated Strategy Code

```python
{{ code }}
```

## Error Details

- **Error Type:** {{ error_type }}
- **Error Message:** {{ error_message }}

## Stack Trace

```
{{ traceback }}
```

## Your Task

1. **Diagnose:** Explain exactly why this code failed. Be specific — reference the exact line or pattern that caused the error.
2. **Root cause:** Identify whether this is a:
   - Syntax error (malformed Python)
   - Import error (forbidden or missing module)
   - Runtime error (division by zero, NaN propagation, wrong column names, index issues)
   - Timeout (infinite loop, excessive computation)
3. **Fix:** Provide the corrected `strategy_candidate.py` code that addresses the specific failure while preserving the original strategy intent.

Output format:

DIAGNOSIS: [1-2 sentences explaining the failure]

CORRECTED CODE:
[Full corrected Python code — raw, no markdown fences]
