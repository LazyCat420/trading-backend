"""
TaskBoard — Async-safe inter-agent communication hub.

Inspired by Claude Code's coordinator/SendMessageTool pattern.
Allows agents to post findings, request investigations, and read
team results during multi-agent debate sessions.

Each ticker+cycle_id gets its own board instance for isolation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Finding:
    """A fact or insight posted by an agent for other agents to see."""

    id: str
    source_agent: str
    content: str
    category: str  # "fact", "risk", "opportunity", "question"
    ticker: str
    confidence: int  # 0-100
    timestamp: float = field(default_factory=time.monotonic)
    responses: list[dict] = field(default_factory=list)


@dataclass
class InvestigationRequest:
    """A request from one agent for another to investigate something."""

    id: str
    requester: str
    target_agent: str  # "*" for any agent
    question: str
    ticker: str
    status: str = "open"  # open, claimed, completed
    claimed_by: str | None = None
    result: str | None = None
    timestamp: float = field(default_factory=time.monotonic)


class TaskBoard:
    """Central hub for inter-agent communication within a debate session.

    Thread-safe via asyncio.Lock. Each board is scoped to a single
    ticker+cycle_id combination.
    """

    def __init__(self):
        self._findings: dict[str, list[Finding]] = {}  # board_key → findings
        self._investigations: dict[str, list[InvestigationRequest]] = {}
        self._lock = asyncio.Lock()
        self._seq = 0

        # Optional: callback for WebSocket broadcast
        self._broadcast_callback: Any = None

    def set_broadcast_callback(self, callback):
        """Set a callback for broadcasting TaskBoard events to the frontend."""
        self._broadcast_callback = callback

    def _board_key(self, ticker: str, cycle_id: str = "") -> str:
        return f"{ticker}:{cycle_id}" if cycle_id else ticker

    async def post_finding(
        self,
        source_agent: str,
        content: str,
        ticker: str,
        cycle_id: str = "",
        category: str = "fact",
        confidence: int = 75,
    ) -> str:
        """Post a finding for other agents to see.

        Returns the finding ID.
        """
        async with self._lock:
            self._seq += 1
            finding_id = f"f-{self._seq:04d}"
            board_key = self._board_key(ticker, cycle_id)

            finding = Finding(
                id=finding_id,
                source_agent=source_agent,
                content=content,
                category=category,
                ticker=ticker,
                confidence=confidence,
            )

            if board_key not in self._findings:
                self._findings[board_key] = []
            self._findings[board_key].append(finding)

            logger.info(
                "[TaskBoard] %s posted finding %s (%s): %s",
                source_agent,
                finding_id,
                category,
                content[:80],
            )

            # Broadcast to frontend if callback is set
            if self._broadcast_callback:
                try:
                    await self._broadcast_callback(
                        {
                            "type": "taskboard_finding",
                            "finding_id": finding_id,
                            "source_agent": source_agent,
                            "category": category,
                            "content": content[:200],
                            "ticker": ticker,
                        }
                    )
                except Exception as e:
                    logger.debug("[TaskBoard] Broadcast failed: %s", e)

            return finding_id

    async def get_findings(
        self,
        ticker: str,
        cycle_id: str = "",
        category: str | None = None,
        exclude_agent: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Get findings for a ticker, optionally filtered by category."""
        async with self._lock:
            board_key = self._board_key(ticker, cycle_id)
            findings = self._findings.get(board_key, [])

            results = []
            for f in findings:
                if category and f.category != category:
                    continue
                if exclude_agent and f.source_agent == exclude_agent:
                    continue
                results.append(
                    {
                        "id": f.id,
                        "source_agent": f.source_agent,
                        "content": f.content,
                        "category": f.category,
                        "confidence": f.confidence,
                        "responses": f.responses,
                    }
                )

            return results[-limit:]

    async def request_investigation(
        self,
        requester: str,
        question: str,
        ticker: str,
        cycle_id: str = "",
        target_agent: str = "*",
    ) -> str:
        """Request another agent to investigate a question.

        Returns the investigation request ID.
        """
        async with self._lock:
            self._seq += 1
            req_id = f"inv-{self._seq:04d}"
            board_key = self._board_key(ticker, cycle_id)

            req = InvestigationRequest(
                id=req_id,
                requester=requester,
                target_agent=target_agent,
                question=question,
                ticker=ticker,
            )

            if board_key not in self._investigations:
                self._investigations[board_key] = []
            self._investigations[board_key].append(req)

            logger.info(
                "[TaskBoard] %s requested investigation %s → %s: %s",
                requester,
                req_id,
                target_agent,
                question[:80],
            )

            return req_id

    async def claim_investigation(
        self,
        req_id: str,
        claiming_agent: str,
        ticker: str,
        cycle_id: str = "",
    ) -> bool:
        """Claim an open investigation request."""
        async with self._lock:
            board_key = self._board_key(ticker, cycle_id)
            for req in self._investigations.get(board_key, []):
                if req.id == req_id and req.status == "open":
                    if req.target_agent != "*" and req.target_agent != claiming_agent:
                        return False
                    req.status = "claimed"
                    req.claimed_by = claiming_agent
                    logger.info(
                        "[TaskBoard] %s claimed investigation %s",
                        claiming_agent,
                        req_id,
                    )
                    return True
            return False

    async def complete_investigation(
        self,
        req_id: str,
        result: str,
        ticker: str,
        cycle_id: str = "",
    ) -> bool:
        """Complete an investigation with results."""
        async with self._lock:
            board_key = self._board_key(ticker, cycle_id)
            for req in self._investigations.get(board_key, []):
                if req.id == req_id and req.status == "claimed":
                    req.status = "completed"
                    req.result = result
                    logger.info(
                        "[TaskBoard] Investigation %s completed: %s",
                        req_id,
                        result[:80],
                    )
                    return True
            return False

    async def get_open_investigations(
        self,
        ticker: str,
        cycle_id: str = "",
        for_agent: str | None = None,
    ) -> list[dict]:
        """Get open investigation requests, optionally filtered for a specific agent."""
        async with self._lock:
            board_key = self._board_key(ticker, cycle_id)
            results = []
            for req in self._investigations.get(board_key, []):
                if req.status != "open":
                    continue
                if (
                    for_agent
                    and req.target_agent != "*"
                    and req.target_agent != for_agent
                ):
                    continue
                results.append(
                    {
                        "id": req.id,
                        "requester": req.requester,
                        "target_agent": req.target_agent,
                        "question": req.question,
                    }
                )
            return results

    def clear_board(self, ticker: str, cycle_id: str = ""):
        """Clear all findings and investigations for a completed cycle."""
        board_key = self._board_key(ticker, cycle_id)
        self._findings.pop(board_key, None)
        self._investigations.pop(board_key, None)
        logger.info("[TaskBoard] Cleared board for %s", board_key)


# Global singleton
task_board = TaskBoard()
