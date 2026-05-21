"""
Pipeline Diagnostics — Phase 1: Audit Queue State and Routing.

Checks:
  1. vLLM endpoint reachability (Jetson, DGX Spark, DGX Spark 2)
  2. Prism Gateway health
  3. Recent LLM audit logs from PostgreSQL
  4. Endpoint call distribution over last 24 hours
  5. Recent cycle audit events (Glance SKIPs, cache hits)

Usage:
    python diagnose_pipeline.py
"""

import asyncio
import sys
import os
import time

# Add project root to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Endpoint Health Check ──────────────────────────────────────────────
async def check_vllm_endpoint(name: str, url: str) -> dict:
    """Probe a vLLM endpoint for model info and KV cache metrics."""
    import httpx

    result = {"name": name, "url": url, "status": "UNKNOWN", "models": [], "cache_pct": 0.0}
    if not url:
        result["status"] = "NOT_CONFIGURED"
        return result

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check /v1/models
            r = await client.get(f"{url}/v1/models")
            if r.status_code == 200:
                models = r.json().get("data", [])
                result["models"] = [m.get("id", "?") for m in models]
                result["status"] = "UP"
            else:
                result["status"] = f"ERROR_{r.status_code}"

            # Check /metrics for KV cache
            try:
                mr = await client.get(f"{url}/metrics")
                if mr.status_code == 200:
                    for line in mr.text.splitlines():
                        if line.startswith("vllm_gpu_cache_usage_perc"):
                            parts = line.split()
                            if len(parts) >= 2:
                                result["cache_pct"] = float(parts[-1]) * 100
            except Exception:
                pass

    except Exception as e:
        result["status"] = f"DOWN ({type(e).__name__})"

    return result


# ── Prism Health Check ─────────────────────────────────────────────────
async def check_prism(url: str, enabled: bool, routing: bool) -> dict:
    """Check Prism Gateway connectivity."""
    import httpx

    result = {"url": url, "enabled": enabled, "routing": routing, "status": "UNKNOWN"}
    if not enabled:
        result["status"] = "DISABLED"
        return result

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{url}/health")
            result["status"] = "UP" if r.status_code == 200 else f"ERROR_{r.status_code}"
    except Exception as e:
        result["status"] = f"DOWN ({type(e).__name__})"

    return result


# ── Database Checks ────────────────────────────────────────────────────
def query_recent_llm_logs() -> list[dict]:
    """Pull last 10 LLM audit logs from PostgreSQL."""
    try:
        from app.db.connection import get_db

        with get_db() as db:
            db.execute("""
                SELECT created_at, ticker, agent_step, endpoint_name, tokens_used
                FROM llm_audit_logs
                ORDER BY created_at DESC
                LIMIT 10
            """)
            rows = db.fetchall()
            return [
                {
                    "time": str(r[0]),
                    "ticker": r[1] or "",
                    "agent": r[2] or "",
                    "endpoint": r[3] or "N/A",
                    "tokens": r[4] or 0,
                }
                for r in rows
            ]
    except Exception as e:
        return [{"error": str(e)}]


def query_endpoint_distribution() -> list[dict]:
    """Count LLM calls per endpoint in the last 24 hours."""
    try:
        from app.db.connection import get_db

        with get_db() as db:
            db.execute("""
                SELECT endpoint_name, COUNT(*) as call_count
                FROM llm_audit_logs
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY endpoint_name
                ORDER BY call_count DESC
            """)
            rows = db.fetchall()
            return [{"endpoint": r[0] or "NULL", "calls_24h": r[1]} for r in rows]
    except Exception as e:
        return [{"error": str(e)}]


def query_recent_audit_events() -> list[dict]:
    """Pull the last 10 cycle audit log entries."""
    try:
        from app.db.connection import get_db

        with get_db() as db:
            db.execute("""
                SELECT timestamp, ticker, event_type, message
                FROM cycle_audit_log
                ORDER BY timestamp DESC
                LIMIT 10
            """)
            rows = db.fetchall()
            return [
                {
                    "time": str(r[0]),
                    "ticker": r[1] or "",
                    "event": r[2] or "",
                    "message": str(r[3])[:120] if r[3] else "",
                }
                for r in rows
            ]
    except Exception as e:
        return [{"error": str(e)}]


def query_glance_skip_count() -> int:
    """Count how many Glance SKIPs happened in the last 24 hours."""
    try:
        from app.db.connection import get_db

        with get_db() as db:
            db.execute("""
                SELECT COUNT(*)
                FROM cycle_audit_log
                WHERE event_type ILIKE '%%glance%%'
                  AND timestamp > NOW() - INTERVAL '24 hours'
            """)
            row = db.fetchone()
            return row[0] if row else 0
    except Exception:
        return -1


# ── Main ───────────────────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("  TRADING PIPELINE DIAGNOSTICS (Phase 1)")
    print("=" * 60)

    # Load settings
    try:
        from app.config import settings
    except Exception as e:
        print(f"\n❌ Failed to load settings: {e}")
        print("   Make sure you have a .env file in the project root.")
        return

    # ── 1. Endpoint Health ─────────────────────────────────────────────
    print("\n┌─ 1. VLLM ENDPOINT HEALTH ─────────────────────────────┐")
    endpoints = [
        ("Jetson", settings.JETSON_VLLM_URL),
        ("DGX Spark", settings.DGX_SPARK_VLLM_URL),
        ("DGX Spark 2", settings.DGX_SPARK_2_VLLM_URL),
    ]
    for name, url in endpoints:
        result = await check_vllm_endpoint(name, url)
        status_icon = "✅" if result["status"] == "UP" else "❌"
        print(f"  {status_icon} {name:15s} | {result['status']:20s} | Models: {result['models']}")
        if result["cache_pct"] > 0:
            cache_icon = "🟡" if result["cache_pct"] > 80 else "🟢"
            print(f"     {cache_icon} KV Cache: {result['cache_pct']:.1f}%")
    print("└───────────────────────────────────────────────────────┘")

    # ── 2. Prism Gateway ───────────────────────────────────────────────
    print("\n┌─ 2. PRISM GATEWAY ────────────────────────────────────┐")
    prism = await check_prism(settings.PRISM_URL, settings.PRISM_ENABLED, settings.PRISM_AGENT_ROUTING)
    status_icon = "✅" if prism["status"] == "UP" else ("⚠️" if prism["status"] == "DISABLED" else "❌")
    print(f"  {status_icon} Status:  {prism['status']}")
    print(f"     Enabled: {prism['enabled']}  |  Agent Routing: {prism['routing']}")
    if not prism["routing"]:
        print(f"     ⚠️  PRISM_AGENT_ROUTING=False → All requests go DIRECT to vLLM")
        print(f"        Prism only receives offline shadow logs (if they succeed)")
    print("└───────────────────────────────────────────────────────┘")

    # ── 3. Recent LLM Activity ─────────────────────────────────────────
    print("\n┌─ 3. LAST 10 LLM CALLS ────────────────────────────────┐")
    logs = query_recent_llm_logs()
    if not logs or (logs and "error" in logs[0]):
        print(f"  ❌ {logs[0].get('error', 'No data') if logs else 'No data'}")
    else:
        for log in logs:
            print(f"  [{log['time'][:19]}] {log['ticker']:6s} | {log['agent']:20s} | {log['endpoint'] or 'N/A':12s} | {log['tokens']} tok")
    print("└───────────────────────────────────────────────────────┘")

    # ── 4. Endpoint Distribution ───────────────────────────────────────
    print("\n┌─ 4. CALLS BY ENDPOINT (Last 24h) ────────────────────┐")
    dist = query_endpoint_distribution()
    if not dist or (dist and "error" in dist[0]):
        print(f"  ❌ {dist[0].get('error', 'No data') if dist else 'No data'}")
    else:
        total = sum(d["calls_24h"] for d in dist)
        for d in dist:
            bar = "█" * min(int(d["calls_24h"] / max(total, 1) * 30), 30)
            print(f"  {d['endpoint']:15s} | {d['calls_24h']:5d} calls | {bar}")
        print(f"  {'TOTAL':15s} | {total:5d} calls")
    print("└───────────────────────────────────────────────────────┘")

    # ── 5. Recent Audit Events ─────────────────────────────────────────
    print("\n┌─ 5. RECENT CYCLE AUDIT EVENTS ────────────────────────┐")
    events = query_recent_audit_events()
    if not events or (events and "error" in events[0]):
        print(f"  ❌ {events[0].get('error', 'No data') if events else 'No data'}")
    else:
        for e in events:
            print(f"  [{e['time'][:19]}] {e['ticker']:6s} | {e['event']:20s} | {e['message'][:60]}")
    print("└───────────────────────────────────────────────────────┘")

    # ── 6. Glance Skip Count ───────────────────────────────────────────
    glance_count = query_glance_skip_count()
    print(f"\n  📊 Glance SKIPs in last 24h: {glance_count}")
    if glance_count > 50:
        print("     ⚠️  High skip count — the pipeline may be over-caching")

    # ── 7. Diagnosis Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DIAGNOSIS SUMMARY")
    print("=" * 60)

    issues = []

    # Check if all endpoints are down
    for name, url in endpoints:
        result = await check_vllm_endpoint(name, url)
        if result["status"] != "UP" and result["status"] != "NOT_CONFIGURED":
            issues.append(f"Endpoint '{name}' is {result['status']}")
        if result["cache_pct"] > 90:
            issues.append(f"Endpoint '{name}' KV cache is critically full ({result['cache_pct']:.0f}%)")

    # Check Prism
    if prism["enabled"] and prism["status"] != "UP":
        issues.append("Prism is enabled but unhealthy — shadow logs may be silently failing")
    if not prism["routing"]:
        issues.append("PRISM_AGENT_ROUTING=False — Prism proxy is bypassed, only shadow logging")

    # Check activity
    if not dist or (dist and "error" in dist[0]):
        issues.append("No LLM calls in the last 24 hours — pipeline may be completely stuck")
    elif total == 0:
        issues.append("Zero LLM calls in the last 24 hours")

    if not issues:
        print("  ✅ No critical issues detected")
    else:
        for issue in issues:
            print(f"  ⚠️  {issue}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
