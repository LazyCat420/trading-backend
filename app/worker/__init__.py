"""
Distributed worker module for tier-based task processing.

Each worker pulls tasks from Redis queues up to its configured max_tier,
executing them against a local vLLM endpoint. Results escalate through
tiers or land in the results queue for the orchestrator.
"""
