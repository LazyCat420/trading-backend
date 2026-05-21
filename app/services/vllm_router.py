import logging
import asyncio
from typing import Any, List, Dict, Optional, Tuple
from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.vllm_client import llm, Priority
from app.services import bot_manager
from app.collectors import congress_scanner, fund_scanner
from app.trading import order_triggers, strategy_tracker

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Pydantic Request Models ──

class ChatRequest(BaseModel):
    system: str
    user: str
    temperature: float = 0.3
    max_tokens: int = 1024
    enable_thinking: bool = False
    priority: int = 1  # Priority.NORMAL
    agent_name: str = "unknown"
    ticker: str = ""
    cycle_id: str = ""
    bot_id: str = ""
    model_override: Optional[str] = None
    endpoint_override: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None


class ChatStreamRequest(BaseModel):
    system: str
    user: str
    temperature: float = 0.3
    max_tokens: int = 8000
    enable_thinking: bool = False
    agent_name: str = "user_chat"
    ticker: str = ""
    model_override: Optional[str] = None
    endpoint_override: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[str]] = None
    bypass_prism: bool = False


class ChatWithToolsRequest(BaseModel):
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    temperature: float = 0.3
    max_tokens: int = 1024
    enable_thinking: bool = False
    priority: int = 1  # Priority.NORMAL
    agent_name: str = "unknown"
    ticker: str = ""
    cycle_id: str = ""
    bot_id: str = ""
    model_override: Optional[str] = None
    endpoint_override: Optional[str] = None


class ConfigureEndpointRequest(BaseModel):
    name: str
    enabled: Optional[bool] = None
    role: Optional[str] = None


class SwitchModelRequest(BaseModel):
    model: str


class UpdateLimitsRequest(BaseModel):
    concurrency_limits: Dict[str, int]


class CacheEndpointRequest(BaseModel):
    model: str
    endpoint_name: str


class CreateBotRequest(BaseModel):
    display_name: str
    starting_cash: float = 100000.0
    description: str = ""


class UpdateBotRequest(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    starting_cash: Optional[float] = None


class SetActiveRequest(BaseModel):
    bot_id: str


class TriggerCreate(BaseModel):
    bot_id: str
    ticker: str
    trigger_type: str
    trigger_price: float
    action: str = "SELL"
    qty_pct: float = 1.0
    trailing_pct: Optional[float] = None
    reason: Optional[str] = None
    created_by: str = "user"


# ── vLLM Dispatching Endpoints ──

@router.post("/api/v1/vllm/chat")
async def vllm_chat(req: ChatRequest):
    try:
        response_text, total_tokens, elapsed_ms = await llm.chat(
            system=req.system,
            user=req.user,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            enable_thinking=req.enable_thinking,
            priority=Priority(req.priority),
            agent_name=req.agent_name,
            ticker=req.ticker,
            cycle_id=req.cycle_id,
            bot_id=req.bot_id,
            model_override=req.model_override,
            endpoint_override=req.endpoint_override,
            history=req.history,
            images=req.images,
            tools=req.tools
        )
        return {"text": response_text, "total_tokens": total_tokens, "elapsed_ms": elapsed_ms}
    except Exception as e:
        logger.exception("Error in /vllm/chat")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/vllm/chat_stream")
async def vllm_chat_stream(req: ChatStreamRequest):
    async def event_generator():
        try:
            async for chunk in llm.chat_stream(
                system=req.system,
                user=req.user,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                enable_thinking=req.enable_thinking,
                agent_name=req.agent_name,
                ticker=req.ticker,
                model_override=req.model_override,
                endpoint_override=req.endpoint_override,
                history=req.history,
                tools=req.tools,
                images=req.images,
                bypass_prism=req.bypass_prism
            ):
                yield chunk + "\n"
        except Exception as e:
            logger.exception("Error in /vllm/chat_stream generator")
            yield f"ERROR: {str(e)}\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/v1/vllm/chat_with_tools")
async def vllm_chat_with_tools(req: ChatWithToolsRequest):
    try:
        result = await llm.chat_with_tools(
            messages=req.messages,
            tools=req.tools,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            enable_thinking=req.enable_thinking,
            priority=Priority(req.priority),
            agent_name=req.agent_name,
            ticker=req.ticker,
            cycle_id=req.cycle_id,
            bot_id=req.bot_id,
            model_override=req.model_override,
            endpoint_override=req.endpoint_override
        )
        return result
    except Exception as e:
        logger.exception("Error in /vllm/chat_with_tools")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/vllm/health")
async def vllm_health():
    try:
        res = await llm.health()
        return {"status": "ok" if res else "unhealthy", "health": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/vllm/health_all")
async def vllm_health_all():
    try:
        res = await llm.health_all()
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/vllm/model")
def vllm_active_model():
    try:
        return {"model": llm.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/vllm/models")
async def vllm_models():
    try:
        models = await llm.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/vllm/role_info")
def vllm_role_info():
    try:
        return llm.get_role_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/vllm/queue_status")
def vllm_queue_status():
    try:
        return llm.queue_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/vllm/endpoints")
def vllm_endpoints():
    try:
        endpoints_data = {}
        for name, ep in llm._endpoints.items():
            endpoints_data[name] = {
                "name": ep.name,
                "url": ep.url,
                "role": ep.role,
                "max_concurrent": ep.max_concurrent,
                "purpose": ep.purpose,
                "enabled": ep.enabled,
                "auto_disabled": ep.auto_disabled,
                "loading": ep.loading,
                "model": ep.model,
                "max_model_len": ep.max_model_len,
                "active_count": ep.active_count,
                "cache_usage": ep.cache_usage,
                "requests_running": ep.requests_running,
                "requests_waiting": ep.requests_waiting,
            }
        return endpoints_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/vllm/configure_endpoint")
def vllm_configure_endpoint(req: ConfigureEndpointRequest):
    try:
        res = llm.configure_endpoint(
            name=req.name,
            enabled=req.enabled,
            role=req.role
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/v1/vllm/rediscover")
async def vllm_rediscover():
    try:
        await llm.rediscover_endpoints()
        return {"status": "rediscovery_triggered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/vllm/switch_model")
def vllm_switch_model(req: SwitchModelRequest):
    try:
        llm.model = req.model
        return {"status": "model_switched", "model": req.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/vllm/update_limits")
def vllm_update_limits(req: UpdateLimitsRequest):
    try:
        for model_id, max_concurrency in req.concurrency_limits.items():
            # Update endpoints with that model
            for ep in llm._endpoints.values():
                if ep.model == model_id:
                    ep.max_concurrent = max_concurrency
                    ep.slots = asyncio.Semaphore(max_concurrency)
                    ep.pipeline_slots = asyncio.Semaphore(max(1, max_concurrency - 1))
        return {"status": "limits_updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/vllm/cache_endpoint")
def vllm_cache_endpoint(req: CacheEndpointRequest):
    try:
        llm._model_endpoint_cache[req.model] = req.endpoint_name
        return {"status": "cache_updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Discovery Scanners Endpoints ──

@router.get("/api/v1/discovery/congress")
def discovery_congress(days: int = 30, min_members: int = 2):
    try:
        res = congress_scanner.find_consensus_trades(days=days, min_members=min_members)
        return res
    except Exception as e:
        logger.exception("Error in /discovery/congress")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/discovery/funds")
def discovery_funds(min_funds: int = 3):
    try:
        res = fund_scanner.find_crossfund_consensus(min_funds=min_funds)
        return res
    except Exception as e:
        logger.exception("Error in /discovery/funds")
        raise HTTPException(status_code=500, detail=str(e))


# ── Bot Profile Management Endpoints ──

@router.get("/api/v1/bot/active_id")
def bot_active_id():
    try:
        return {"bot_id": bot_manager.get_active_bot_id()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/bot/starting_cash/{bot_id}")
def bot_starting_cash(bot_id: str):
    try:
        return {"starting_cash": bot_manager.get_bot_starting_cash(bot_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/bot/description/{bot_id}")
def bot_description(bot_id: str):
    try:
        return {"description": bot_manager.get_bot_description(bot_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/bot/active")
def bot_set_active(req: SetActiveRequest):
    try:
        bot_manager.set_active_bot(req.bot_id)
        return {"bot_id": req.bot_id, "switched": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/bot/profiles")
def bot_profiles():
    try:
        return bot_manager.list_bot_profiles()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/bot/profiles")
def bot_create_profile(req: CreateBotRequest):
    try:
        res = bot_manager.create_bot_profile(
            display_name=req.display_name,
            starting_cash=req.starting_cash,
            description=req.description
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/v1/bot/profiles/{bot_id}")
def bot_update_profile(bot_id: str, req: UpdateBotRequest):
    try:
        res = bot_manager.update_bot_profile(
            bot_id=bot_id,
            display_name=req.display_name,
            description=req.description,
            starting_cash=req.starting_cash
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/bot/profiles/{bot_id}/reset")
def bot_reset_profile(bot_id: str):
    try:
        res = bot_manager.reset_bot_profile(bot_id)
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/v1/bot/profiles/{bot_id}")
def bot_delete_profile(bot_id: str):
    try:
        res = bot_manager.delete_bot_profile(bot_id)
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/bot/cycle_running")
def bot_cycle_running():
    try:
        return {"cycle_running": bot_manager.is_cycle_running()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Order Triggers Endpoints ──

@router.get("/api/v1/triggers")
def triggers_list(bot_id: str, active_only: bool = True):
    try:
        res = order_triggers.list_triggers(bot_id, active_only=active_only)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/triggers")
async def triggers_create(req: TriggerCreate):
    try:
        res = await order_triggers.create_trigger(
            bot_id=req.bot_id,
            ticker=req.ticker,
            trigger_type=req.trigger_type,
            trigger_price=req.trigger_price,
            action=req.action,
            qty_pct=req.qty_pct,
            trailing_pct=req.trailing_pct,
            reason=req.reason,
            created_by=req.created_by
        )
        if "error" in res:
            raise HTTPException(status_code=400, detail=res["error"])
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/v1/triggers/{trigger_id}")
async def triggers_cancel(trigger_id: str):
    try:
        res = await order_triggers.cancel_trigger(trigger_id)
        if "error" in res:
            raise HTTPException(status_code=404, detail=res["error"])
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Strategy Tracker Endpoints ──

@router.get("/api/v1/strategies/rankings")
def strategies_rankings(limit: int = 50):
    try:
        res = strategy_tracker.compute_rankings(limit=limit)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/strategies/timeline/{ticker}")
def strategies_timeline(ticker: str, limit: int = 20):
    try:
        res = strategy_tracker.get_ticker_strategy_timeline(ticker=ticker, limit=limit)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/strategies/bench-underperformers")
def strategies_bench_underperformers():
    try:
        res = strategy_tracker.bench_underperformers()
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
