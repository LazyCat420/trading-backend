"""
Coordination Tools — Agent-callable tools for inter-agent communication.

These tools wrap the TaskBoard to let agents share findings, request
investigations, and read team insights during swarm debates.

Inspired by Claude Code's SendMessageTool and TeamCreateTool patterns.
"""

import json
import logging

from app.tools.registry import registry, PermissionLevel
from app.agents.task_board import task_board

logger = logging.getLogger(__name__)


# ── Tool 1: Post Finding ──────────────────────────────────────────────
@registry.register(
    name="post_finding",
    description=(
        "Share a discovered fact, risk, or opportunity with other agents on the team. "
        "Other agents can see your findings and use them to strengthen or challenge arguments. "
        "Categories: 'fact' (verified data point), 'risk' (potential danger), "
        "'opportunity' (potential upside), 'question' (open question for the team)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The finding to share (e.g. 'RSI is at 28.5 — stock is oversold').",
            },
            "ticker": {
                "type": "string",
                "description": "The stock ticker this finding relates to.",
            },
            "category": {
                "type": "string",
                "enum": ["fact", "risk", "opportunity", "question"],
                "description": "Type of finding.",
            },
            "confidence": {
                "type": "integer",
                "description": "How confident you are in this finding (0-100).",
            },
        },
        "required": ["content", "ticker", "category"],
    },
    tier=1,
    source="coordination",
    permission=PermissionLevel.WRITE,
    tags=["coordinate", "share", "finding", "team", "debate"],
)
async def post_finding(
    content: str,
    ticker: str,
    category: str = "fact",
    confidence: int = 75,
    _agent_name: str = "unknown_agent",
    _cycle_id: str = "",
) -> str:
    """Post a finding to the TaskBoard for other agents to see."""
    try:
        finding_id = await task_board.post_finding(
            source_agent=_agent_name,
            content=content,
            ticker=ticker,
            cycle_id=_cycle_id,
            category=category,
            confidence=confidence,
        )
        return json.dumps(
            {
                "status": "posted",
                "finding_id": finding_id,
                "message": f"Finding shared with the team: {content[:60]}...",
            }
        )
    except Exception as e:
        logger.exception("[CoordinationTools] post_finding failed")
        return json.dumps({"status": "error", "message": str(e)})


# ── Tool 2: Read Team Findings ─────────────────────────────────────────
@registry.register(
    name="read_team_findings",
    description=(
        "Read all findings posted by other agents on the team for a specific ticker. "
        "Use this to see what your teammates have discovered before formulating "
        "your argument. You can filter by category (fact/risk/opportunity/question)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker to read findings for.",
            },
            "category": {
                "type": "string",
                "enum": ["fact", "risk", "opportunity", "question"],
                "description": "Optional: filter by finding type.",
            },
        },
        "required": ["ticker"],
    },
    tier=0,
    source="coordination",
    tags=["coordinate", "read", "findings", "team"],
)
async def read_team_findings(
    ticker: str,
    category: str | None = None,
    _agent_name: str = "unknown_agent",
    _cycle_id: str = "",
) -> str:
    """Read findings from the TaskBoard, excluding your own posts."""
    try:
        findings = await task_board.get_findings(
            ticker=ticker,
            cycle_id=_cycle_id,
            category=category,
            exclude_agent=_agent_name,
        )

        if not findings:
            return json.dumps(
                {
                    "status": "success",
                    "ticker": ticker,
                    "findings": [],
                    "message": "No findings from other agents yet.",
                }
            )

        return json.dumps(
            {
                "status": "success",
                "ticker": ticker,
                "findings_count": len(findings),
                "findings": findings,
            }
        )
    except Exception as e:
        logger.exception("[CoordinationTools] read_team_findings failed")
        return json.dumps({"status": "error", "message": str(e)})


# ── Tool 3: Request Investigation ──────────────────────────────────────
@registry.register(
    name="request_investigation",
    description=(
        "Ask another agent (or any available agent) to investigate a specific question. "
        "Use this when you need a fact-check, additional data, or a different perspective. "
        "Set target_agent to '*' to let any agent pick it up, or specify an agent name."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to investigate (e.g. 'What is NVDA\\'s current P/E ratio compared to sector average?').",
            },
            "ticker": {
                "type": "string",
                "description": "The stock ticker this investigation relates to.",
            },
            "target_agent": {
                "type": "string",
                "description": "Agent name to investigate, or '*' for any agent. Default '*'.",
            },
        },
        "required": ["question", "ticker"],
    },
    tier=1,
    source="coordination",
    permission=PermissionLevel.WRITE,
    tags=["coordinate", "investigate", "delegate", "question"],
)
async def request_investigation(
    question: str,
    ticker: str,
    target_agent: str = "*",
    _agent_name: str = "unknown_agent",
    _cycle_id: str = "",
) -> str:
    """Request another agent to investigate a question."""
    try:
        # Prevent self-targeting to avoid infinite loops
        if target_agent == _agent_name:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Cannot request investigation from yourself. Pick another agent or '*'.",
                }
            )
            
        # Fallback for hallucinated agent targets
        VALID_AGENTS = {
            "*", "sentiment_agent", "technical_agent", "fundamental_agent",
            "risk_agent", "macro_agent", "valuation_agent", "synthesizer", "planner"
        }
        if target_agent not in VALID_AGENTS:
            target_agent = "*"
            
        req_id = await task_board.request_investigation(
            requester=_agent_name,
            question=question,
            ticker=ticker,
            cycle_id=_cycle_id,
            target_agent=target_agent,
        )
        return json.dumps(
            {
                "status": "requested",
                "investigation_id": req_id,
                "message": f"Investigation requested: {question[:60]}...",
                "target": target_agent,
            }
        )
    except Exception as e:
        logger.exception("[CoordinationTools] request_investigation failed")
        return json.dumps({"status": "error", "message": str(e)})


# ── Tool 4: Check Open Investigations ─────────────────────────────────
@registry.register(
    name="check_open_investigations",
    description=(
        "Check if any other agents have requested investigations that you can help with. "
        "Returns a list of open questions waiting for an answer."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker to check investigations for.",
            },
        },
        "required": ["ticker"],
    },
    tier=0,
    source="coordination",
    tags=["coordinate", "investigations", "open", "help"],
)
async def check_open_investigations(
    ticker: str,
    _agent_name: str = "unknown_agent",
    _cycle_id: str = "",
) -> str:
    """Check for open investigation requests that this agent can help with."""
    try:
        investigations = await task_board.get_open_investigations(
            ticker=ticker,
            cycle_id=_cycle_id,
            for_agent=_agent_name,
        )

        if not investigations:
            return json.dumps(
                {
                    "status": "success",
                    "ticker": ticker,
                    "open_investigations": [],
                    "message": "No open investigations for you.",
                }
            )

        return json.dumps(
            {
                "status": "success",
                "ticker": ticker,
                "investigation_count": len(investigations),
                "open_investigations": investigations,
            }
        )
    except Exception as e:
        logger.exception("[CoordinationTools] check_open_investigations failed")
        return json.dumps({"status": "error", "message": str(e)})
