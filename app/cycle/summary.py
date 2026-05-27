def tally_results(results: list[dict], summary: dict) -> None:
    for r in results:
        action = r.get("action", "HOLD")
        summary[f"{action.lower()}_count"] = summary.get(f"{action.lower()}_count", 0) + 1
        if r.get("human_review"):
            summary["review_count"] = summary.get("review_count", 0) + 1
