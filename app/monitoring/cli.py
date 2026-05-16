"""
CLI Monitor — Terminal-based live vLLM monitoring.

Usage:
    python -m app.monitoring.cli

Shows real-time token flows, latency, and agent activity.
"""

import asyncio
import os
from app.services.vllm_client import llm
from app.monitoring.llm_tracker import tracker
from app.monitoring.metrics_collector import metrics
from app.config import settings


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def format_number(n: int | float) -> str:
    """Format large numbers with commas."""
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


async def collect_and_display():
    """Main monitoring loop — collects metrics and displays dashboard."""
    print(f"Starting vLLM Monitor — {settings.JETSON_VLLM_URL}")
    print("Press Ctrl+C to exit\n")

    # Subscribe to live calls
    queue = tracker.subscribe()
    recent_calls: list[str] = []  # last 15 call summaries

    iteration = 0
    while True:
        # Collect Jetson metrics every 5 iterations (avoid hammering)
        if iteration % 5 == 0:
            await metrics.collect_once()

        # Drain any new calls from the queue
        while not queue.empty():
            try:
                record = queue.get_nowait()
                recent_calls.append(f"  {record.timestamp[11:19]} {record.summary}")
                if len(recent_calls) > 15:
                    recent_calls.pop(0)
            except asyncio.QueueEmpty:
                break

        # Build display
        stats = tracker.get_stats()
        agent_stats = tracker.get_agent_stats()
        latest = metrics.get_latest() or {}

        healthy = await llm.health()
        health_icon = "🟢 HEALTHY" if healthy else "🔴 OFFLINE"

        clear_screen()

        # Header
        print(f"{'═' * 66}")
        print("  vLLM Monitor — Multi-Endpoint")
        print(
            f"  Active Model: {llm.model[:60] if llm.model else 'Auto-discovering...'}"
        )
        print(f"  Status: {health_icon}")
        print(
            f"  Total calls: {format_number(stats['total_calls'])} "
            f"({stats['failed_calls']} failed)"
        )
        print(f"{'═' * 66}")

        # Token stats
        print(
            f"  Tokens In:  {format_number(stats['total_prompt_tokens']):>10s}"
            f"  │  Avg: {format_number(stats['avg_prompt_tokens']):>7s}/call"
        )
        print(
            f"  Tokens Out: {format_number(stats['total_completion_tokens']):>10s}"
            f"  │  Avg: {format_number(stats['avg_completion_tokens']):>7s}/call"
        )
        print(
            f"  Total:      {format_number(stats['total_tokens']):>10s}"
            f"  │  Avg latency: {format_number(stats['avg_latency_ms'])}ms"
        )

        # Jetson metrics
        if latest:
            print(f"{'─' * 66}")
            print(
                f"  Jetson  │  Requests: {latest.get('num_requests_running', 0):.0f} running"
                f"  {latest.get('num_requests_waiting', 0):.0f} waiting"
            )
            print(
                f"          │  KV Cache: {latest.get('gpu_cache_usage_pct', 0):.1f}%"
                f"  │  Throughput: {latest.get('avg_generation_throughput', 0):.1f} tok/s"
            )
            print(
                f"          │  Latency:  p50={latest.get('e2e_latency_p50', 0):.2f}s"
                f"  p95={latest.get('e2e_latency_p95', 0):.2f}s"
                f"  p99={latest.get('e2e_latency_p99', 0):.2f}s"
            )

        # Semaphore — dynamic for all endpoints
        print(f"{'─' * 66}")
        q_status = llm.queue_status()

        for ep_name, ep_data in q_status.items():
            if ep_name in (
                "reserved_for_chat",
                "dgx",
            ):  # skip meta keys and compat alias
                continue
            if not isinstance(ep_data, dict) or "active" not in ep_data:
                continue
            active = ep_data["active"]
            max_c = ep_data["max_concurrent"]
            bar = "█" * min(active, max_c) + "░" * max(0, max_c - active)
            label = ep_name.replace("_", " ").title()
            model_str = (
                f" ({ep_data.get('model', '?')[:30]})" if ep_data.get("model") else ""
            )
            print(f"  {label:20s} [{bar}] {active}/{max_c}{model_str}")

        # Adaptive Concurrency Controller
        try:
            from app.services.adaptive_concurrency import concurrency_controller
            cc = concurrency_controller.status()
            print(f"{'─' * 66}")
            print(
                f"  Concurrency Ctrl   │  Limit: {cc['current_limit']} "
                f"(range {cc['min']}–{cc['max']})  │  Cache: {cc['cache_avg_pct']:.0f}%"
            )
            print(
                f"                     │  vLLM running: {cc['total_running_on_vllm']}  "
                f"waiting: {cc['total_waiting_on_vllm']}  "
                f"capacity: {cc['total_capacity']}"
            )
            if cc.get("per_label"):
                labels = ", ".join(
                    f"{k}: {v}" for k, v in cc["per_label"].items()
                )
                print(f"                     │  Active: {labels}")
        except Exception:
            pass

        # Per-agent stats
        if agent_stats:
            print(f"{'─' * 66}")
            print(f"  {'Agent':<20s} {'Calls':>6s} {'Tokens':>8s} {'Avg ms':>8s}")
            print(f"  {'─' * 20} {'─' * 6} {'─' * 8} {'─' * 8}")
            for agent, s in sorted(agent_stats.items()):
                print(
                    f"  {agent:<20s} {s['calls']:>6d} "
                    f"{s['total_tokens']:>8d} {s['avg_latency_ms']:>7.0f}ms"
                )

        # Recent calls
        if recent_calls:
            print(f"{'═' * 66}")
            print("  LIVE CALLS:")
            for line in recent_calls[-10:]:
                print(line)

        print(f"{'═' * 66}")
        print("  Refreshing every 2s... (Ctrl+C to exit)")

        iteration += 1
        await asyncio.sleep(2)


async def main():
    try:
        await collect_and_display()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
    finally:
        pass  # cleanup handled by GC
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
