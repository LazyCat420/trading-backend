import json
import logging
from datetime import datetime, timezone
import uuid
from app.db.connection import get_db

logger = logging.getLogger(__name__)

def init_trace_table():
    """Ensure the llm_traces table exists."""
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS llm_traces (
                id VARCHAR(64) PRIMARY KEY,
                cycle_id VARCHAR(64),
                bot_id VARCHAR(64),
                ticker VARCHAR(16),
                agent_name VARCHAR(64),
                system_prompt TEXT,
                user_prompt TEXT,
                response_text TEXT,
                temperature FLOAT,
                tokens INT,
                elapsed_ms INT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_traces_cycle ON llm_traces(cycle_id);
            CREATE INDEX IF NOT EXISTS idx_traces_agent ON llm_traces(agent_name);
        """)

def log_trace(
    cycle_id: str,
    bot_id: str,
    ticker: str,
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    response_text: str,
    temperature: float,
    tokens: int,
    elapsed_ms: int
):
    """Save an exact replica of the LLM call for replay and debugging."""
    try:
        with get_db() as db:
            db.execute("""
                INSERT INTO llm_traces (
                    id, cycle_id, bot_id, ticker, agent_name, 
                    system_prompt, user_prompt, response_text, 
                    temperature, tokens, elapsed_ms, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, [
                str(uuid.uuid4()),
                cycle_id,
                bot_id,
                ticker,
                agent_name,
                system_prompt,
                user_prompt,
                response_text,
                temperature,
                tokens,
                elapsed_ms,
                datetime.now(timezone.utc)
            ])
    except Exception as e:
        logger.debug("[TraceStore] Failed to save trace: %s", e)

# Initialize on import
try:
    init_trace_table()
except Exception as e:
    logger.debug("Failed to init trace table: %s", e)
