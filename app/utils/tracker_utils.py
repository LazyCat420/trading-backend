def get_prior_quarter(quarter: str) -> str | None:
    if not quarter or len(quarter) < 6:
        return None
    try:
        year, q = int(quarter[:4]), int(quarter[-1])
        return f"{year - 1}Q4" if q == 1 else f"{year}Q{q - 1}"
    except (ValueError, IndexError):
        return None


def calculate_qoq_change(current_shares: int, prior_shares: int, is_new: bool) -> str:
    if is_new:
        return "NEW"
    if current_shares == prior_shares:
        return "UNCHANGED"
    if current_shares > prior_shares:
        return "ADDED"
    return "REDUCED"


def calculate_trend_direction(history: list[tuple[str, int]]) -> tuple[str, int, float]:
    """
    Returns (direction, streak, total_change_pct)
     history expects a list of (quarter, shares) tuples, chronologically ordered earliest to latest.
    """
    if len(history) <= 1:
        return "NEW", 0, 0.0

    streak = 0
    direction = "STEADY"

    # Run through history backwards to compute trend streak
    for i in range(len(history) - 1, 0, -1):
        prev_shares = history[i - 1][1]
        curr_shares = history[i][1]
        delta = curr_shares - prev_shares

        if i == len(history) - 1:
            direction = (
                "ACCUMULATING" if delta > 0 else "DUMPING" if delta < 0 else "STEADY"
            )
            streak = 1
        elif (direction == "ACCUMULATING" and delta > 0) or (
            direction == "DUMPING" and delta < 0
        ):
            streak += 1
        else:
            break

    first_shares = history[0][1]
    curr_shares = history[-1][1]
    total_change_pct = (
        round(((curr_shares - first_shares) / first_shares) * 100, 1)
        if first_shares > 0
        else 0.0
    )

    return direction, streak, total_change_pct
