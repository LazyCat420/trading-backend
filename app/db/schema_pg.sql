-- ══════════════════════════════════════════
-- MARKET DATA
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS price_history (
    ticker      TEXT,
    date        DATE,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    source      TEXT,
    PRIMARY KEY (ticker, date, source)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker              TEXT,
    snapshot_date       DATE,
    source              TEXT DEFAULT 'yfinance',
    market_cap          DOUBLE PRECISION,
    pe_ratio            DOUBLE PRECISION,
    forward_pe          DOUBLE PRECISION,
    peg_ratio           DOUBLE PRECISION,
    price_to_book       DOUBLE PRECISION,
    price_to_sales      DOUBLE PRECISION,
    ev_to_ebitda        DOUBLE PRECISION,
    profit_margin       DOUBLE PRECISION,
    roe                 DOUBLE PRECISION,
    roa                 DOUBLE PRECISION,
    revenue             DOUBLE PRECISION,
    revenue_growth      DOUBLE PRECISION,
    net_income          DOUBLE PRECISION,
    debt_to_equity      DOUBLE PRECISION,
    current_ratio       DOUBLE PRECISION,
    beta                DOUBLE PRECISION,
    week_52_high        DOUBLE PRECISION,
    week_52_low         DOUBLE PRECISION,
    short_float_pct     DOUBLE PRECISION,
    PRIMARY KEY (ticker, snapshot_date)
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    ticker              TEXT,
    fetched_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    data_source         TEXT,
    candles_used        INTEGER,
    price               DOUBLE PRECISION,
    open                DOUBLE PRECISION,
    high                DOUBLE PRECISION,
    low                 DOUBLE PRECISION,
    volume              BIGINT,
    vwap                DOUBLE PRECISION,
    rsi_14              DOUBLE PRECISION,
    macd                DOUBLE PRECISION,
    macd_signal         DOUBLE PRECISION,
    macd_hist           DOUBLE PRECISION,
    bb_upper            DOUBLE PRECISION,
    bb_lower            DOUBLE PRECISION,
    bb_pct              DOUBLE PRECISION,
    sma_20              DOUBLE PRECISION,
    sma_50              DOUBLE PRECISION,
    sma_200             DOUBLE PRECISION,
    atr_14              DOUBLE PRECISION,
    adx_14              DOUBLE PRECISION,
    stoch_k             DOUBLE PRECISION,
    stoch_d             DOUBLE PRECISION,
    returns_1d          DOUBLE PRECISION,
    returns_5d          DOUBLE PRECISION,
    returns_20d         DOUBLE PRECISION,
    volatility_20d      DOUBLE PRECISION,
    sharpe_20d          DOUBLE PRECISION,
    max_drawdown_20d    DOUBLE PRECISION,
    beta_20d            DOUBLE PRECISION,
    pe_ratio            DOUBLE PRECISION,
    forward_pe          DOUBLE PRECISION,
    eps                 DOUBLE PRECISION,
    market_cap          DOUBLE PRECISION,
    revenue_growth      DOUBLE PRECISION,
    profit_margin       DOUBLE PRECISION,
    debt_to_equity      DOUBLE PRECISION,
    PRIMARY KEY (ticker, fetched_at)
);

CREATE TABLE IF NOT EXISTS technicals (
    ticker          TEXT,
    date            DATE,
    rsi_14          DOUBLE PRECISION,
    macd            DOUBLE PRECISION,
    macd_signal     DOUBLE PRECISION,
    macd_hist       DOUBLE PRECISION,
    sma_20          DOUBLE PRECISION,
    sma_50          DOUBLE PRECISION,
    sma_200         DOUBLE PRECISION,
    ema_12          DOUBLE PRECISION,
    ema_26          DOUBLE PRECISION,
    bb_upper        DOUBLE PRECISION,
    bb_mid          DOUBLE PRECISION,
    bb_lower        DOUBLE PRECISION,
    atr_14          DOUBLE PRECISION,
    adx_14          DOUBLE PRECISION,
    stoch_k         DOUBLE PRECISION,
    stoch_d         DOUBLE PRECISION,
    obv             DOUBLE PRECISION,
    vwap            DOUBLE PRECISION,
    support         DOUBLE PRECISION,
    resistance      DOUBLE PRECISION,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS financial_history (
    ticker              TEXT,
    period_type         TEXT,
    period_end          DATE,
    revenue             DOUBLE PRECISION,
    gross_profit        DOUBLE PRECISION,
    operating_income    DOUBLE PRECISION,
    net_income          DOUBLE PRECISION,
    eps                 DOUBLE PRECISION,
    free_cash_flow      DOUBLE PRECISION,
    PRIMARY KEY (ticker, period_type, period_end)
);

CREATE TABLE IF NOT EXISTS balance_sheet (
    ticker              TEXT,
    period_end          DATE,
    total_assets        DOUBLE PRECISION,
    total_liabilities   DOUBLE PRECISION,
    total_equity        DOUBLE PRECISION,
    cash                DOUBLE PRECISION,
    total_debt          DOUBLE PRECISION,
    working_capital     DOUBLE PRECISION,
    PRIMARY KEY (ticker, period_end)
);

-- ══════════════════════════════════════════
-- ASSET PRICES (consolidated)
-- Replaces: international_prices, futures_data,
--           commodities, crypto_prices
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS asset_prices (
    symbol      TEXT,
    asset_class TEXT,
    exchange    TEXT DEFAULT '',
    date        DATE,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      DOUBLE PRECISION,
    currency    TEXT DEFAULT 'USD',
    source      TEXT DEFAULT 'openbb',
    PRIMARY KEY (symbol, asset_class, date)
);

-- ══════════════════════════════════════════
-- MACRO DATA
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS macro_indicators (
    indicator   TEXT,
    date        DATE,
    value       DOUBLE PRECISION,
    country     TEXT DEFAULT 'US',
    source      TEXT DEFAULT 'openbb',
    PRIMARY KEY (indicator, date, country)
);

-- ══════════════════════════════════════════
-- ALT DATA (sentiment/social)
-- Separate tables — each source has different
-- fields, keeps DBeaver debugging simple
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS news_articles (
    id              TEXT PRIMARY KEY,
    ticker          TEXT,
    title           TEXT,
    publisher       TEXT,
    url             TEXT,
    published_at    TIMESTAMP,
    summary         TEXT,
    llm_summary     TEXT,
    source          TEXT,
    summarized_at   TIMESTAMP,
    quality_status  TEXT,
    quality_reason  TEXT,
    quality_score   INTEGER,
    screenshot      TEXT,
    cluster_id      TEXT,
    is_cluster_winner BOOLEAN,
    collected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scraper_scripts (
    domain          TEXT PRIMARY KEY,
    script          TEXT,
    script_type     TEXT,
    success_count   INTEGER DEFAULT 0,
    fail_count      INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'active',
    last_success    TIMESTAMP,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reddit_posts (
    id                  TEXT PRIMARY KEY,
    ticker              TEXT,
    subreddit           TEXT,
    title               TEXT,
    body                TEXT,
    score               INTEGER,
    upvote_ratio        DOUBLE PRECISION,
    comment_count       INTEGER,
    flair               TEXT,
    sentiment_score     DOUBLE PRECISION,
    award_count         INTEGER,
    comment_velocity    DOUBLE PRECISION,
    summary             TEXT,
    created_utc         TIMESTAMP,
    summarized_at       TIMESTAMP,
    quality_status      TEXT,
    quality_reason      TEXT,
    quality_score       INTEGER,
    collected_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS youtube_transcripts (
    video_id        TEXT PRIMARY KEY,
    ticker          TEXT,
    title           TEXT,
    channel         TEXT,
    raw_transcript  TEXT,
    summary         TEXT,
    tickers_mentioned TEXT,
    thumbnail_url   TEXT,
    published_at    TIMESTAMP,
    duration_secs   INTEGER,
    summarized_at   TIMESTAMP,
    quality_status  TEXT,
    quality_reason  TEXT,
    quality_score   INTEGER,
    collected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS data_archive (
    id              SERIAL PRIMARY KEY,
    source_table    TEXT,
    source_id       TEXT,
    ticker          TEXT,
    title           TEXT,
    content         TEXT,
    original_date   TIMESTAMP,
    purge_after     TIMESTAMP,
    archived_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_table, source_id)
);

-- ══════════════════════════════════════════
-- INSTITUTIONAL & CONGRESS
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS sec_13f_holdings (
    cik            TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    name_of_issuer TEXT,
    cusip          TEXT,
    value_usd      DOUBLE PRECISION,
    shares         BIGINT,
    share_type     TEXT,
    pct_change     DOUBLE PRECISION,
    is_new_position BOOLEAN,
    is_exit        BOOLEAN,
    filing_quarter TEXT NOT NULL,
    filing_date    DATE,
    collected_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (cik, ticker, filing_quarter)
);

CREATE TABLE IF NOT EXISTS congress_trades (
    id                  TEXT PRIMARY KEY,
    politician          TEXT,
    party               TEXT,
    chamber             TEXT,
    state               TEXT,
    ticker              TEXT,
    transaction_type    TEXT,
    amount_range        TEXT,
    trade_date          DATE,
    disclosure_date     DATE,
    days_to_disclose    INTEGER
);

CREATE TABLE IF NOT EXISTS fund_alerts (
    id              TEXT PRIMARY KEY,
    created_at      TIMESTAMP,
    alert_type      TEXT,
    ticker          TEXT,
    entity_name     TEXT,
    detail          TEXT,
    severity        TEXT,
    llm_summary     TEXT,
    is_read         BOOLEAN DEFAULT FALSE
);

-- ══════════════════════════════════════════
-- VECTOR STORE (unified RAG search)
-- ══════════════════════════════════════════
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
    id              TEXT PRIMARY KEY,
    source_table    TEXT,
    source_id       TEXT,
    ticker          TEXT,
    content_preview TEXT,
    embedding       vector(384),
    created_at      TIMESTAMP
);

CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
ON embeddings USING hnsw (embedding vector_cosine_ops);

-- ══════════════════════════════════════════
-- USER BUCKET (uploaded files)
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS user_data (
    id              TEXT PRIMARY KEY,
    filename        TEXT,
    file_type       TEXT,
    raw_content     TEXT,
    processed_at    TIMESTAMP,
    tags            TEXT,
    embedding       vector(384)
);

-- ══════════════════════════════════════════
-- TRADING & SCHEDULING
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS cycle_schedules (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    schedule_type   TEXT NOT NULL,         -- 'cron' | 'interval'
    cron_expression TEXT,
    interval_hours  DOUBLE PRECISION,
    collect         BOOLEAN DEFAULT TRUE,
    "analyze"       BOOLEAN DEFAULT TRUE,
    trade           BOOLEAN,                  -- NULL = use armed state
    tickers         TEXT,                  -- JSONB array
    max_tickers     INTEGER,                  -- NULL = use .env default
    market_hours_only BOOLEAN DEFAULT FALSE,
    is_active       BOOLEAN DEFAULT TRUE,
    last_run_at     TIMESTAMP,
    next_run_at     TIMESTAMP,
    run_count       INTEGER DEFAULT 0,
    last_status     TEXT,
    last_error      TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watchlist (
    ticker        TEXT PRIMARY KEY,
    status        TEXT DEFAULT 'active',   -- active | paused | removed | banned
    status_reason TEXT,
    banned_at     TIMESTAMP,
    added_at      TIMESTAMP,
    source        TEXT,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    id              TEXT PRIMARY KEY,
    bot_id          TEXT,
    ticker          TEXT,
    qty             DOUBLE PRECISION,
    avg_entry_price DOUBLE PRECISION,
    stop_loss_pct   DOUBLE PRECISION DEFAULT 0.08,
    opened_at       TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,
    bot_id          TEXT,
    ticker          TEXT,
    side            TEXT,
    qty             DOUBLE PRECISION,
    price           DOUBLE PRECISION,
    signal          TEXT,
    created_at      TIMESTAMP,
    filled_at       TIMESTAMP,
    realized_pnl    DOUBLE PRECISION
);

-- ══════════════════════════════════════════
-- BROKER LEDGER (lot-level accounting)
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS trade_fills (
    fill_id         TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL,
    bot_id          TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    side            TEXT NOT NULL,
    fill_qty        DOUBLE PRECISION NOT NULL,
    fill_price      DOUBLE PRECISION NOT NULL,
    fill_value      DOUBLE PRECISION NOT NULL,
    fees            DOUBLE PRECISION DEFAULT 0.0,
    filled_at       TIMESTAMP NOT NULL,
    cycle_id        TEXT,
    source          TEXT DEFAULT 'pipeline'
);

CREATE TABLE IF NOT EXISTS position_lots (
    lot_id          TEXT PRIMARY KEY,
    bot_id          TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    fill_id         TEXT NOT NULL,
    opened_at       TIMESTAMP NOT NULL,
    original_qty    DOUBLE PRECISION NOT NULL,
    remaining_qty   DOUBLE PRECISION NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    status          TEXT DEFAULT 'open',
    cycle_id        TEXT,
    is_legacy       BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS lot_closures (
    closure_id      TEXT PRIMARY KEY,
    bot_id          TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    sell_fill_id    TEXT NOT NULL,
    lot_id          TEXT NOT NULL,
    closed_qty      DOUBLE PRECISION NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION NOT NULL,
    realized_pnl    DOUBLE PRECISION NOT NULL,
    closed_at       TIMESTAMP NOT NULL,
    holding_days    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_fills_bot ON trade_fills(bot_id);
CREATE INDEX IF NOT EXISTS idx_fills_ticker ON trade_fills(ticker);
CREATE INDEX IF NOT EXISTS idx_fills_order ON trade_fills(order_id);
CREATE INDEX IF NOT EXISTS idx_lots_bot ON position_lots(bot_id);
CREATE INDEX IF NOT EXISTS idx_lots_ticker ON position_lots(ticker);
CREATE INDEX IF NOT EXISTS idx_lots_status ON position_lots(status);
CREATE INDEX IF NOT EXISTS idx_closures_lot ON lot_closures(lot_id);
CREATE INDEX IF NOT EXISTS idx_closures_ticker ON lot_closures(ticker);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id              TEXT PRIMARY KEY,
    bot_id          TEXT,
    snapshot_ts     TIMESTAMP,
    cash_balance    DOUBLE PRECISION,
    total_value     DOUBLE PRECISION,
    realized_pnl    DOUBLE PRECISION,
    unrealized_pnl  DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS price_triggers (
    id              TEXT PRIMARY KEY,
    bot_id          TEXT,
    ticker          TEXT,
    trigger_type    TEXT,
    price           DOUBLE PRECISION,
    active          BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decision_outcomes (
    id            TEXT PRIMARY KEY,
    cycle_id      TEXT,
    ticker        TEXT,
    action        TEXT,
    confidence    INTEGER,
    entry_price   DOUBLE PRECISION,
    exit_price    DOUBLE PRECISION,
    pnl_pct       DOUBLE PRECISION,
    outcome       TEXT,
    lesson_stored TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at   TIMESTAMP
);

-- ══════════════════════════════════════════
-- BOTS & LLM AUDIT
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS bots (
    bot_id          TEXT PRIMARY KEY,
    display_name    TEXT,
    model_name      TEXT,
    status          TEXT DEFAULT 'idle',
    cash_balance    DOUBLE PRECISION DEFAULT 100000.0,
    starting_cash   DOUBLE PRECISION DEFAULT 100000.0,
    total_pnl       DOUBLE PRECISION DEFAULT 0.0,
    win_rate        DOUBLE PRECISION DEFAULT 0.0,
    total_trades    INTEGER DEFAULT 0,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP,
    last_run_at     TIMESTAMP,
    description     TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS context_blobs (
    context_hash    TEXT PRIMARY KEY,
    content         TEXT,
    byte_size       INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_audit_logs (
    id                  TEXT PRIMARY KEY,
    cycle_id            TEXT,
    bot_id              TEXT,
    ticker              TEXT,
    agent_step          TEXT,
    model               TEXT,
    system_prompt_hash  TEXT,
    context_hash        TEXT,
    raw_response        TEXT,
    tokens_used         INTEGER,
    execution_ms        INTEGER,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    endpoint_name       TEXT,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    queue_wait_ms       INTEGER,
    tokens_per_second   DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS analysis_results (
    id          TEXT PRIMARY KEY,
    cycle_id    TEXT,
    bot_id      TEXT,
    ticker      TEXT,
    agent_name  TEXT,
    result_json TEXT,
    confidence  INTEGER,
    created_at  TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decision_evaluations (
    decision_id        TEXT PRIMARY KEY,  -- fk to llm_audit_logs.id
    cycle_id           TEXT,
    ticker             TEXT,
    timestamp          TIMESTAMP,
    difficulty         TEXT,
    evidence_gathering TEXT,
    policy_understanding BOOLEAN,
    first_principles_reasoning TEXT,
    prompt_snapshot_link TEXT,
    raw_output_link    TEXT,
    red_cards          TEXT,              -- JSONB array of red cards
    judge_a_score      DOUBLE PRECISION,
    judge_b_score      DOUBLE PRECISION,
    discrepancy_trigger BOOLEAN DEFAULT FALSE,
    final_quality_score DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS discovered_tickers (
    ticker          TEXT,
    source          TEXT,
    context         TEXT,
    score           DOUBLE PRECISION,
    discovered_at   TIMESTAMP,
    PRIMARY KEY (ticker, source)
);

CREATE TABLE IF NOT EXISTS scheduler_history (
    id          TEXT PRIMARY KEY,
    job_name    TEXT,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP,
    status      TEXT,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS flash_briefings (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    report_content  TEXT,
    source_urls     TEXT[],
    article_count   INTEGER
);

CREATE TABLE IF NOT EXISTS morning_briefings (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    report_content  TEXT,
    tickers_evaluated TEXT[]
);

-- ══════════════════════════════════════════
-- DATA SOURCE HEALTH (fallback tracking)
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS data_source_status (
    source          TEXT,
    ticker          TEXT,
    last_success    TIMESTAMP,
    last_failure    TIMESTAMP,
    rows_fetched    INTEGER DEFAULT 0,
    error_msg       TEXT,
    PRIMARY KEY (source, ticker)
);

-- ══════════════════════════════════════════
-- AUTORESEARCH PERSISTENCE
-- ══════════════════════════════════════════

CREATE TABLE IF NOT EXISTS autoresearch_reports (
    id                      TEXT PRIMARY KEY,
    cycle_id                TEXT,
    status                  TEXT,
    phase                   TEXT,
    error                   TEXT,
    data_gaps               TEXT,
    decision_issues         TEXT,
    llm_issues              TEXT,
    data_quality_score      DOUBLE PRECISION,
    decision_quality_score  DOUBLE PRECISION,
    llm_performance_score   DOUBLE PRECISION,
    performance_metrics     TEXT,
    reflection              TEXT,
    recovery_stats          TEXT,
    overall_score           DOUBLE PRECISION,
    created_at              TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_autoresearch_reports_cycle ON autoresearch_reports(cycle_id);

CREATE TABLE IF NOT EXISTS autoresearch_cycle_summaries (
    id              TEXT PRIMARY KEY,
    cycle_id        TEXT UNIQUE,
    total_tickers   INTEGER,
    buy_count       INTEGER,
    sell_count      INTEGER,
    hold_count      INTEGER,
    avg_confidence  DOUBLE PRECISION,
    top_ticker      TEXT,
    top_confidence  INTEGER,
    lesson_summary  TEXT,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS execution_errors (
    id              TEXT PRIMARY KEY,
    cycle_id        TEXT,
    phase           TEXT,
    ticker          TEXT,
    error_type      TEXT,
    error_message   TEXT,
    stack_trace     TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cycle_audit_log (
    id TEXT PRIMARY KEY,
    cycle_id TEXT,
    timestamp TIMESTAMPTZ,
    audit_type TEXT,
    event_type TEXT,
    phase TEXT,
    ticker TEXT,
    severity TEXT,
    message TEXT,
    data JSONB
);
CREATE INDEX IF NOT EXISTS idx_cycle_audit_log_cycle ON cycle_audit_log(cycle_id);

-- ══════════════════════════════════════════
-- RAG A/B TESTING
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS rag_ab_results (
    id              TEXT PRIMARY KEY,
    ticker          TEXT,
    query           TEXT,
    strategy        TEXT,
    chunks_returned INTEGER,
    top_score       DOUBLE PRECISION,
    avg_score       DOUBLE PRECISION,
    retrieval_ms    INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rag_ab_strategy ON rag_ab_results(strategy);
CREATE INDEX IF NOT EXISTS idx_rag_ab_ticker ON rag_ab_results(ticker);

-- ══════════════════════════════════════════
-- USER FEEDBACK (unified notes + constraints)
-- Replaces: user_notes, user_constraints
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS user_feedback (
    id              TEXT PRIMARY KEY,
    ticker          TEXT,
    feedback_type   TEXT,   -- note | constraint | thesis | signal | ban_reason | flag_reason
    content         TEXT,
    sentiment       TEXT,   -- bullish | bearish | neutral (nullable)
    confidence      INTEGER,   -- 0-100 (nullable)
    constraint_type TEXT,   -- no_sell | max_position | min_confidence (nullable)
    constraint_val  TEXT,   -- constraint value (nullable)
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at      TIMESTAMP,
    is_active       BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_user_feedback_ticker ON user_feedback(ticker);
CREATE INDEX IF NOT EXISTS idx_user_feedback_type ON user_feedback(feedback_type);

-- ══════════════════════════════════════════
-- TICKER BAN SYSTEM
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS ticker_bans (
    ticker        TEXT PRIMARY KEY,
    reason        TEXT,
    ban_type      TEXT DEFAULT 'manual',   -- manual | auto | pattern
    pattern_tags  TEXT,                    -- JSONB: ["sub_penny", "no_volume"]
    market_cap    DOUBLE PRECISION,
    price_at_ban  DOUBLE PRECISION,
    volume_at_ban BIGINT,
    banned_by     TEXT DEFAULT 'user',
    banned_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ══════════════════════════════════════════
-- DATA QUALITY FLAGS
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS data_flags (
    id           TEXT PRIMARY KEY,
    source_table TEXT,    -- news_articles | reddit_posts | youtube_transcripts
    source_id    TEXT,
    ticker       TEXT,
    flag_type    TEXT,    -- spam | clickbait | irrelevant | outdated | fake
    reason       TEXT,
    flagged_by   TEXT DEFAULT 'user',
    flagged_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    auto_action  TEXT     -- excluded | deleted | source_warned
);

CREATE INDEX IF NOT EXISTS idx_data_flags_source ON data_flags(source_table, source_id);
CREATE INDEX IF NOT EXISTS idx_data_flags_ticker ON data_flags(ticker);

-- ══════════════════════════════════════════
-- SOURCE TRUST SCORES
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS source_trust (
    source_type  TEXT,    -- publisher | subreddit | youtube_channel
    source_name  TEXT,
    trust_score  DOUBLE PRECISION DEFAULT 1.0,  -- 1.0 = trusted, 0.0 = blocked
    total_flags  INTEGER DEFAULT 0,
    total_items  INTEGER DEFAULT 0,
    flag_rate    DOUBLE PRECISION DEFAULT 0.0,
    quality_wins INTEGER DEFAULT 0,
    win_rate     DOUBLE PRECISION DEFAULT 0.0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_type, source_name)
);

-- ══════════════════════════════════════════
-- BAN PATTERNS (learned from user bans)
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS ban_patterns (
    id           TEXT PRIMARY KEY,
    pattern_name TEXT,
    conditions   TEXT,     -- JSONB: {"price_lt": 0.50, "volume_lt": 10000}
    source_bans  INTEGER DEFAULT 0,
    auto_filter  BOOLEAN DEFAULT TRUE,  -- opt-in by default
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id              TEXT PRIMARY KEY,
    title           TEXT DEFAULT 'New Chat',
    created_at      TIMESTAMP,
    ended_at        TIMESTAMP,
    message_count   INTEGER DEFAULT 0,
    is_active       BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS chat_history (
    id              TEXT PRIMARY KEY,
    session_id      TEXT,
    ticker          TEXT,
    user_message    TEXT,
    bot_response    TEXT,
    context_hash    TEXT,
    model_used      TEXT,
    tokens_used     INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_ticker ON chat_history(ticker);

-- ══════════════════════════════════════════
-- PRISM MEMORY SYSTEM
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS semantic_memory (
    id               TEXT PRIMARY KEY,
    ticker           TEXT,
    type             TEXT,     -- fact | rule | preference | threshold
    content          TEXT,
    confidence       DOUBLE PRECISION DEFAULT 0.5,
    source_agent     TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count     INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_semantic_memory_ticker ON semantic_memory(ticker);

CREATE TABLE IF NOT EXISTS episodic_memory (
    id                 TEXT PRIMARY KEY,
    cycle_id           TEXT,
    ticker             TEXT,
    timestamp          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    summary            TEXT,
    key_decisions      TEXT,       -- JSONB array
    outcome            TEXT,       -- positive/negative/neutral
    outcome_score      DOUBLE PRECISION,        -- -1.0 to 1.0
    agents_involved    TEXT        -- JSONB array
);
CREATE INDEX IF NOT EXISTS idx_episodic_memory_ticker ON episodic_memory(ticker);

CREATE TABLE IF NOT EXISTS procedural_memory (
    id                 TEXT PRIMARY KEY,
    ticker             TEXT,
    trigger_pattern    TEXT,
    procedure          TEXT,       -- JSONB array of steps
    success_count      INTEGER DEFAULT 0,
    failure_count      INTEGER DEFAULT 0,
    success_rate       DOUBLE PRECISION DEFAULT 0.0,
    last_triggered_at  TIMESTAMP,
    created_by_agent   TEXT
);
CREATE INDEX IF NOT EXISTS idx_procedural_memory_ticker ON procedural_memory(ticker);

CREATE TABLE IF NOT EXISTS prospective_memory (
    id                 TEXT PRIMARY KEY,
    ticker             TEXT,
    intention          TEXT,
    trigger_condition  TEXT,
    priority           TEXT,      -- critical/high/medium/low
    status             TEXT,      -- pending/triggered/expired
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    trigger_at         TIMESTAMP,
    context            TEXT
);
CREATE INDEX IF NOT EXISTS idx_prospective_memory_ticker ON prospective_memory(ticker);
CREATE INDEX IF NOT EXISTS idx_prospective_memory_status ON prospective_memory(status);

-- ══════════════════════════════════════════
-- AUTORESEARCH EXPERIENCES (Reflector Loop)
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS agent_experiences (
    id              TEXT PRIMARY KEY,
    agent_name      TEXT NOT NULL,
    task_context    TEXT NOT NULL,
    lesson_learned  TEXT NOT NULL,
    success_score   DOUBLE PRECISION DEFAULT 1.0,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_applied    TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_agent_experiences_name ON agent_experiences(agent_name);

-- ══════════════════════════════════════════
-- CAPSULE CONTEXT (Layer 2 expandable storage)
-- Stores full raw agent responses for lazy-fetch expansion.
-- Layer 1 (compressed summaries) travel in-memory via AgentCapsule.
-- Layer 2 (full data) stays here, accessible via get_cycle_context tool.
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS cycle_context (
    id              TEXT PRIMARY KEY,         -- UUID
    cycle_id        TEXT NOT NULL,
    agent_name      TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    raw_response    TEXT NOT NULL,            -- Full agent output (Layer 2)
    summary         TEXT,                     -- Capsule Layer 1 summary
    signal          TEXT,                     -- BUY | SELL | HOLD | NEUTRAL | UNKNOWN
    confidence      DOUBLE PRECISION,         -- 0.0–1.0
    flags           TEXT,                     -- JSON array of flags
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_cycle_context_cycle ON cycle_context(cycle_id);
CREATE INDEX IF NOT EXISTS idx_cycle_context_ticker ON cycle_context(ticker);
CREATE INDEX IF NOT EXISTS idx_cycle_context_agent ON cycle_context(agent_name);

-- ══════════════════════════════════════════
-- INDEXES
-- ══════════════════════════════════════════
CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_articles(ticker);
CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_reddit_ticker ON reddit_posts(ticker);
-- ══════════════════════════════════════════
-- YOUTUBE CHANNEL MANAGEMENT
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS youtube_channels (
    channel_handle  TEXT PRIMARY KEY,
    display_name    TEXT,
    added_by        TEXT DEFAULT 'system',
    is_active       BOOLEAN DEFAULT TRUE,
    total_videos    INTEGER DEFAULT 0,
    last_scraped    TIMESTAMP,
    added_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS discovered_channels (
    channel_handle  TEXT PRIMARY KEY,
    display_name    TEXT,
    discovery_count INTEGER DEFAULT 1,
    avg_view_count  DOUBLE PRECISION,
    first_seen      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status          TEXT DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_congress_ticker ON congress_trades(ticker);
CREATE INDEX IF NOT EXISTS idx_congress_trade_date ON congress_trades(trade_date);
CREATE INDEX IF NOT EXISTS idx_13f_ticker ON sec_13f_holdings(ticker);
CREATE INDEX IF NOT EXISTS idx_13f_quarter ON sec_13f_holdings(filing_quarter);
CREATE INDEX IF NOT EXISTS idx_alerts_ticker ON fund_alerts(ticker);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON fund_alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_analysis_cycle ON analysis_results(cycle_id);
CREATE INDEX IF NOT EXISTS idx_analysis_ticker ON analysis_results(ticker);
CREATE INDEX IF NOT EXISTS idx_audit_cycle ON llm_audit_logs(cycle_id);
CREATE INDEX IF NOT EXISTS idx_audit_ticker ON llm_audit_logs(ticker);
CREATE INDEX IF NOT EXISTS idx_audit_context ON llm_audit_logs(context_hash);
CREATE INDEX IF NOT EXISTS idx_embed_ticker ON embeddings(ticker);
CREATE INDEX IF NOT EXISTS idx_embed_source ON embeddings(source_table);
CREATE INDEX IF NOT EXISTS idx_asset_class ON asset_prices(asset_class);
CREATE INDEX IF NOT EXISTS idx_positions_bot ON positions(bot_id);
CREATE INDEX IF NOT EXISTS idx_orders_bot ON orders(bot_id);

-- ══════════════════════════════════════════
-- KNOWLEDGE GRAPH & ONTOLOGY
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS ontology_nodes (
    id                  TEXT PRIMARY KEY,
    node_type           TEXT,
    label               TEXT,
    activation          DOUBLE PRECISION DEFAULT 0.0,
    embedding           vector(384),
    metadata_json       TEXT,
    validated_count     INTEGER DEFAULT 0,
    contradicted_count  INTEGER DEFAULT 0,
    disproven           BOOLEAN DEFAULT FALSE,
    source_cycle_id     TEXT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ontology_edges (
    id                  TEXT PRIMARY KEY,
    source_id           TEXT,
    target_id           TEXT,
    relation            TEXT,
    weight              DOUBLE PRECISION,
    decay               DOUBLE PRECISION,
    evidence_count      INTEGER DEFAULT 1,
    metadata_json       TEXT,
    source_cycle_id     TEXT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ontology_nodes_type ON ontology_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_ontology_edges_source ON ontology_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_ontology_edges_target ON ontology_edges(target_id);

CREATE TABLE IF NOT EXISTS company_registry (
    symbol        TEXT PRIMARY KEY,
    company_name  TEXT NOT NULL,
    aliases       TEXT,
    sector        TEXT,
    market_cap    DOUBLE PRECISION DEFAULT 0,
    is_sp500      BOOLEAN DEFAULT FALSE,
    verified      BOOLEAN DEFAULT FALSE,
    rejected      BOOLEAN DEFAULT FALSE,
    source        TEXT DEFAULT 'sp500_load',
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ticker_metadata (
    ticker          TEXT PRIMARY KEY,
    name            TEXT,
    sector          TEXT,
    industry        TEXT,
    market_cap      BIGINT,
    market_cap_tier TEXT,  -- mega/large/mid/small/micro
    asset_class     TEXT,  -- stock/crypto/commodity/etf
    sp500           BOOLEAN DEFAULT FALSE,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ticker_correlations (
    ticker_a    TEXT,
    ticker_b    TEXT,
    correlation DOUBLE PRECISION,
    tier        TEXT,     -- highly_correlated/correlated/weakly_correlated/inversely_correlated
    period      TEXT,     -- '30d' or '90d'
    data_points INTEGER,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker_a, ticker_b, period)
);

CREATE INDEX IF NOT EXISTS idx_corr_a ON ticker_correlations(ticker_a);
CREATE INDEX IF NOT EXISTS idx_corr_b ON ticker_correlations(ticker_b);
CREATE INDEX IF NOT EXISTS idx_meta_sector ON ticker_metadata(sector);

-- ══════════════════════════════════════════
-- SECTOR DASHBOARD
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS sector_performance (
    sector              TEXT,
    date                DATE,
    avg_return_1d       DOUBLE PRECISION,
    avg_return_5d       DOUBLE PRECISION,
    avg_return_30d      DOUBLE PRECISION,
    avg_return_60d      DOUBLE PRECISION,
    avg_return_6mo      DOUBLE PRECISION,
    avg_return_1y       DOUBLE PRECISION,
    relative_strength_1y DOUBLE PRECISION,
    breadth_pct         DOUBLE PRECISION,
    top_gainer          TEXT,
    top_gainer_return   DOUBLE PRECISION,
    top_loser           TEXT,
    top_loser_return    DOUBLE PRECISION,
    avg_volume_ratio    DOUBLE PRECISION,
    momentum_signal     TEXT,
    stock_count         INTEGER,
    computed_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (sector, date)
);

CREATE TABLE IF NOT EXISTS sector_correlations (
    sector_a    TEXT,
    sector_b    TEXT,
    correlation DOUBLE PRECISION,
    tier        TEXT,
    period      TEXT,
    data_points INTEGER,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (sector_a, sector_b, period)
);

CREATE TABLE IF NOT EXISTS stock_commodity_correlations (
    ticker                  TEXT,
    commodity               TEXT,
    correlation             DOUBLE PRECISION,
    sensitivity             TEXT,
    period                  TEXT,
    data_points             INTEGER,
    -- Quant methods (Phase 2: advanced correlation engine)
    cointegration_pvalue    DOUBLE PRECISION,          -- Engle-Granger p-value (lower = stronger)
    cointegrated            BOOLEAN,         -- p < 0.05
    lead_lag_days           INTEGER,         -- negative = commodity leads stock
    lead_lag_correlation    DOUBLE PRECISION,          -- correlation at best lag
    vol_adj_correlation     DOUBLE PRECISION,          -- volatility-weighted correlation
    correlation_stability   DOUBLE PRECISION,          -- std dev of rolling correlations (lower = more stable)
    distance_correlation    DOUBLE PRECISION,          -- non-linear dependency (0-1)
    quant_score             INTEGER,         -- composite score (0-100)
    method_details          TEXT,         -- JSONB with full breakdown
    computed_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, commodity, period)
);

CREATE TABLE IF NOT EXISTS sector_rotation_signals (
    id                  TEXT PRIMARY KEY,
    from_sector         TEXT,
    to_sector           TEXT,
    from_return_5d      DOUBLE PRECISION,
    to_return_5d        DOUBLE PRECISION,
    correlation         DOUBLE PRECISION,
    commodity_trigger   TEXT,
    confidence          TEXT,
    evidence_json       TEXT,
    detected_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sector_perf_date ON sector_performance(date);
CREATE INDEX IF NOT EXISTS idx_sector_corr_a ON sector_correlations(sector_a);
CREATE INDEX IF NOT EXISTS idx_stock_comm_ticker ON stock_commodity_correlations(ticker);
CREATE INDEX IF NOT EXISTS idx_rotation_date ON sector_rotation_signals(detected_at);

-- ══════════════════════════════════════════
-- DATA CURATION — blocklist for deleted items
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS deleted_data (
    source          TEXT NOT NULL,     -- 'news', 'reddit', 'youtube'
    item_id         TEXT NOT NULL,     -- primary key from the source table
    title           TEXT,              -- for audit trail
    deleted_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reason          TEXT DEFAULT 'user_delete',
    PRIMARY KEY (source, item_id)
);
CREATE INDEX IF NOT EXISTS idx_deleted_source ON deleted_data(source);

-- ══════════════════════════════════════════
-- COLLABORATION & CONFIGURATION
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS user_notes (
    id              TEXT PRIMARY KEY,
    ticker          TEXT,
    note_type       TEXT,
    content         TEXT,
    sentiment       TEXT,
    confidence      DOUBLE PRECISION,
    is_active       BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS user_constraints (
    id              TEXT PRIMARY KEY,
    ticker          TEXT,
    constraint_type TEXT,
    value           TEXT,
    reason          TEXT,
    is_active       BOOLEAN DEFAULT TRUE
);

-- NOTE: chat_history already defined above (line ~505) with session_id support
-- NOTE: youtube_channels already defined above (line ~527) with added_at column

CREATE TABLE IF NOT EXISTS rss_feeds (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    url             TEXT NOT NULL UNIQUE,
    is_active       BOOLEAN DEFAULT TRUE,
    added_by        TEXT DEFAULT 'system',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS monitored_subreddits (
    id              TEXT PRIMARY KEY,
    subreddit       TEXT NOT NULL UNIQUE,
    is_active       BOOLEAN DEFAULT TRUE,
    added_by        TEXT DEFAULT 'system',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ══════════════════════════════════════════
-- CROSS-ASSET MARKET INTELLIGENCE
-- ══════════════════════════════════════════

-- NOTE: asset_prices already defined above (line ~98) with exchange + currency columns

-- Per-sector breadth metrics (% above moving averages, new highs/lows)
CREATE TABLE IF NOT EXISTS sector_breadth (
    sector          TEXT,
    date            DATE,
    pct_above_sma50 DOUBLE PRECISION,
    pct_above_sma200 DOUBLE PRECISION,
    new_highs       INTEGER,
    new_lows        INTEGER,
    net_highs       INTEGER,
    computed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (sector, date)
);

-- Market regime snapshots (VIX, yields, dollar, overall regime)
CREATE TABLE IF NOT EXISTS market_regime (
    date                DATE PRIMARY KEY,
    vix_level           DOUBLE PRECISION,
    vix_signal          TEXT,
    vix_zscore          DOUBLE PRECISION,
    vix_term_ratio      DOUBLE PRECISION,
    vix_term_signal     TEXT,
    yield_2y            DOUBLE PRECISION,
    yield_10y           DOUBLE PRECISION,
    yield_2y10y_spread  DOUBLE PRECISION,
    yield_signal        TEXT,
    dollar_index        DOUBLE PRECISION,
    dollar_change_5d    DOUBLE PRECISION,
    sp500_level         DOUBLE PRECISION,
    sp500_change_5d     DOUBLE PRECISION,
    breadth_sp500       DOUBLE PRECISION,
    regime_label        TEXT,
    computed_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Generic cross-asset correlations (sectors ↔ commodities/VIX/yields/dollar)
CREATE TABLE IF NOT EXISTS cross_asset_correlations (
    asset_a         TEXT,
    asset_b         TEXT,
    correlation     DOUBLE PRECISION,
    tier            TEXT,
    period          TEXT,
    data_points     INTEGER,
    computed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (asset_a, asset_b, period)
);

-- ══════════════════════════════════════════
-- CYCLE BENCHMARKS (run-over-run comparison)
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS cycle_benchmarks (
    cycle_id            TEXT PRIMARY KEY,
    started_at          TIMESTAMP,
    finished_at         TIMESTAMP,
    total_ms            INTEGER,
    collect_ms          INTEGER,
    analyze_ms          INTEGER,
    trade_ms            INTEGER,
    ticker_count        INTEGER,
    avg_ticker_ms       INTEGER,
    steps_total         INTEGER,
    steps_skipped       INTEGER,
    steps_ok            INTEGER,
    steps_error         INTEGER,
    total_tokens        INTEGER,
    cache_hit_pct       DOUBLE PRECISION,          -- % of collector steps skipped (cache)
    status              TEXT           -- done | error | stopped
);

CREATE TABLE IF NOT EXISTS cycle_ticker_benchmarks (
    cycle_id            TEXT,
    ticker              TEXT,
    collect_ms          INTEGER,
    analyze_ms          INTEGER,
    total_ms            INTEGER,
    steps_skipped       INTEGER,
    steps_ok            INTEGER,
    tokens_used         INTEGER,
    action              TEXT,          -- BUY | SELL | HOLD
    confidence          INTEGER,
    PRIMARY KEY (cycle_id, ticker)
);

-- ══════════════════════════════════════════════════════════════════
-- GLOBAL DATA SCHEMA
-- Separate PostgreSQL schema for global macro/geopolitical data.
-- Keeps global tables logically isolated from main trading tables
-- while enabling full JOIN capability across schemas.
-- ══════════════════════════════════════════════════════════════════

CREATE SCHEMA IF NOT EXISTS global;

-- ── EIA Energy Data (Weekly Petroleum Reports) ──────────────────
CREATE TABLE IF NOT EXISTS global.energy_reports (
    id              TEXT PRIMARY KEY,
    series_id       TEXT,        -- EIA series: 'PET.WCESTUS1.W'
    indicator       TEXT,        -- 'crude_inventory', 'gasoline_prod', etc.
    date            DATE,
    value           DOUBLE PRECISION,
    unit            TEXT,        -- 'thousand_barrels', 'thousand_bpd'
    source          TEXT DEFAULT 'eia',
    collected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_energy_date ON global.energy_reports(date);
CREATE INDEX IF NOT EXISTS idx_energy_indicator ON global.energy_reports(indicator);

-- ── Conflict Events (GDELT-based, replaces ACLED) ──────────────
CREATE TABLE IF NOT EXISTS global.conflict_events (
    id              TEXT PRIMARY KEY,
    event_id_acled  INTEGER,        -- Legacy column name, 0 for GDELT-sourced events
    event_date      DATE,
    year            INTEGER,
    event_type      TEXT,        -- 'Battles', 'Explosions/Remote violence', 'Protests'
    sub_event_type  TEXT,
    actor1          TEXT,
    actor2          TEXT,
    country         TEXT,
    region          TEXT,        -- 'Middle East', 'Eastern Europe', etc.
    admin1          TEXT,        -- province/state
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    fatalities      INTEGER,
    notes           TEXT,
    source_acled    TEXT,        -- Legacy name, stores source domain for GDELT events
    chokepoint_proximity TEXT,   -- 'hormuz', 'suez', 'bab_el_mandeb', 'none'
    source          TEXT DEFAULT 'gdelt',  -- 'gdelt' (was 'acled')
    collected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conflict_date ON global.conflict_events(event_date);
CREATE INDEX IF NOT EXISTS idx_conflict_region ON global.conflict_events(region);
CREATE INDEX IF NOT EXISTS idx_conflict_country ON global.conflict_events(country);
CREATE INDEX IF NOT EXISTS idx_conflict_chokepoint ON global.conflict_events(chokepoint_proximity);

-- ── UN COMTRADE Trade Flows ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS global.trade_flows (
    id              TEXT PRIMARY KEY,
    reporter_code   INTEGER,
    reporter        TEXT,        -- 'United States', 'China', etc.
    partner_code    INTEGER,
    partner         TEXT,
    commodity_code  TEXT,        -- HS code: '2709' (crude oil)
    commodity_desc  TEXT,
    trade_flow      TEXT,        -- 'Import' or 'Export'
    value_usd       DOUBLE PRECISION,
    net_weight_kg   DOUBLE PRECISION,
    period          TEXT,        -- '202603' (YYYYMM)
    source          TEXT DEFAULT 'comtrade',
    collected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trade_period ON global.trade_flows(period);
CREATE INDEX IF NOT EXISTS idx_trade_commodity ON global.trade_flows(commodity_code);
CREATE INDEX IF NOT EXISTS idx_trade_reporter ON global.trade_flows(reporter);

-- ── Geopolitical Risk Index (Caldara-Iacoviello) ────────────────
CREATE TABLE IF NOT EXISTS global.gpr_index (
    date            DATE PRIMARY KEY,
    gpr             DOUBLE PRECISION,         -- overall GPR index
    gpr_threats     DOUBLE PRECISION,         -- threats sub-index (GPRT)
    gpr_acts        DOUBLE PRECISION,         -- acts sub-index (GPRA)
    source          TEXT DEFAULT 'policyuncertainty',
    collected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Anomaly Detection Results ───────────────────────────────────
CREATE TABLE IF NOT EXISTS global.anomalies (
    id              TEXT PRIMARY KEY,
    data_source     TEXT,        -- 'eia', 'acled', 'comtrade', 'gpr'
    indicator       TEXT,        -- 'crude_inventory_change', 'conflict_count_mideast'
    date            DATE,
    observed_value  DOUBLE PRECISION,
    expected_value  DOUBLE PRECISION,         -- EWMA or rolling mean
    z_score         DOUBLE PRECISION,
    severity        TEXT,        -- 'NORMAL', 'ELEVATED', 'CRITICAL'
    description     TEXT,        -- "Oil inventories drew -6.2M vs expected -1.5M"
    affected_assets TEXT,        -- JSONB: ["XLE","OXY","CL=F","GLD"]
    detected_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_anomaly_date ON global.anomalies(date);
CREATE INDEX IF NOT EXISTS idx_anomaly_source ON global.anomalies(data_source);
CREATE INDEX IF NOT EXISTS idx_anomaly_severity ON global.anomalies(severity);

-- ── Cross-Correlation Snapshots ─────────────────────────────────
CREATE TABLE IF NOT EXISTS global.cross_correlations (
    id              TEXT PRIMARY KEY,
    series_a        TEXT,        -- 'eia_crude_inventory'
    series_b        TEXT,        -- 'acled_mideast_events'
    correlation     DOUBLE PRECISION,         -- Pearson
    granger_pvalue  DOUBLE PRECISION,         -- Granger causality p-value (A→B)
    granger_reverse DOUBLE PRECISION,         -- Granger causality p-value (B→A)
    lead_lag_days   INTEGER,        -- optimal lag (negative = A leads B)
    method          TEXT,        -- 'pearson', 'granger', 'distance'
    window_days     INTEGER,        -- rolling window size
    computed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_xcorr_series ON global.cross_correlations(series_a, series_b);

-- ── Regime Detection Snapshots ──────────────────────────────────
CREATE TABLE IF NOT EXISTS global.regime_states (
    date            DATE PRIMARY KEY,
    regime_label    TEXT,        -- 'risk_off', 'risk_on', 'crisis', 'transition'
    regime_prob     DOUBLE PRECISION,         -- HMM posterior probability
    gpr_level       TEXT,        -- 'LOW', 'MODERATE', 'HIGH', 'EXTREME'
    energy_state    TEXT,        -- 'oversupply', 'balanced', 'deficit'
    trade_state     TEXT,        -- 'expanding', 'stable', 'contracting'
    composite_score DOUBLE PRECISION,         -- 0-100 multi-signal composite
    computed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Intelligence Briefs (LLM-generated summaries for RAG) ───────
CREATE TABLE IF NOT EXISTS global.intelligence_briefs (
    id              TEXT PRIMARY KEY,
    brief_type      TEXT,        -- 'daily_macro', 'energy', 'geopolitical', 'trade', 'composite'
    period_start    DATE,
    period_end      DATE,
    summary         TEXT,        -- LLM-generated intelligence brief
    risk_level      TEXT,        -- 'NORMAL', 'ELEVATED', 'CRITICAL'
    anomaly_count   INTEGER,
    affected_assets TEXT,        -- JSONB: ["XLE","GLD","CL=F"]
    raw_data_count  INTEGER,
    tokens_used     INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_brief_type ON global.intelligence_briefs(brief_type);
CREATE INDEX IF NOT EXISTS idx_brief_date ON global.intelligence_briefs(period_end);

-- ── War News Feed (geo-tagged news for the intelligence map) ────
CREATE TABLE IF NOT EXISTS global.war_news_feed (
    id              TEXT PRIMARY KEY,
    headline        TEXT,
    url             TEXT,
    source_domain   TEXT,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    location_name   TEXT,
    tone            DOUBLE PRECISION,         -- GDELT average tone (-100 to +100)
    themes          TEXT,        -- JSONB array of GDELT themes
    timestamp       TIMESTAMP,
    data_source     TEXT DEFAULT 'gdelt',
    collected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_war_news_ts ON global.war_news_feed(timestamp);
CREATE INDEX IF NOT EXISTS idx_war_news_source ON global.war_news_feed(data_source);

-- ── Tanker Positions (AIS snapshots at maritime chokepoints) ─────
CREATE TABLE IF NOT EXISTS global.tanker_positions (
    id              TEXT PRIMARY KEY,
    mmsi            TEXT,        -- Maritime Mobile Service Identity
    vessel_name     TEXT,
    vessel_type     TEXT,        -- 'Tanker', 'Cargo', etc.
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    speed           DOUBLE PRECISION,         -- knots
    heading         DOUBLE PRECISION,         -- degrees
    destination     TEXT,
    flag            TEXT,        -- country flag
    zone            TEXT,        -- 'hormuz', 'bab_el_mandeb', 'suez', 'malacca'
    timestamp       TIMESTAMP,
    collected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tanker_zone ON global.tanker_positions(zone);
CREATE INDEX IF NOT EXISTS idx_tanker_ts ON global.tanker_positions(timestamp);

-- ── Chokepoint Alerts (derived from tanker + conflict proximity) ──
CREATE TABLE IF NOT EXISTS global.chokepoint_alerts (
    id              TEXT PRIMARY KEY,
    zone            TEXT,        -- 'hormuz', 'suez', 'bab_el_mandeb', 'malacca'
    alert_level     TEXT,        -- 'NORMAL', 'ELEVATED', 'CRITICAL'
    tanker_count    INTEGER,
    nearby_conflict_count INTEGER,
    avg_tanker_speed DOUBLE PRECISION,        -- low speed = congestion
    reroute_count   INTEGER,        -- tankers avoiding the zone
    war_news_count  INTEGER DEFAULT 0,
    price_impact_score DOUBLE PRECISION,      -- 0-1 estimated oil price impact
    timestamp       TIMESTAMP,
    computed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chokepoint_zone ON global.chokepoint_alerts(zone);
CREATE INDEX IF NOT EXISTS idx_chokepoint_ts ON global.chokepoint_alerts(timestamp);

-- ── COMTRADE Tracked Commodities Reference ──────────────────────
-- Static reference table for HS codes we track
CREATE TABLE IF NOT EXISTS global.tracked_commodities (
    hs_code         TEXT PRIMARY KEY,
    name            TEXT,
    category        TEXT,        -- 'energy', 'metals', 'agriculture', 'livestock', 'soft'
    trading_symbol  TEXT,        -- futures ticker: 'CL=F', 'GC=F', etc.
    is_active       BOOLEAN DEFAULT TRUE
);

-- Seed tracked commodities
INSERT INTO global.tracked_commodities VALUES ('2709',  'Crude Oil',         'energy',      'CL=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('2711',  'Natural Gas & LNG', 'energy',      'NG=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('2701',  'Coal',              'energy',      NULL,    TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('7108',  'Gold',              'metals',      'GC=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('7106',  'Silver',            'metals',      'SI=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('7403',  'Copper',            'metals',      'HG=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('7601',  'Aluminum',          'metals',      'ALI=F', TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('2601',  'Iron Ore',          'metals',      NULL,    TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('7110',  'Platinum',          'metals',      'PL=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('1001',  'Wheat',             'agriculture', 'ZW=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('1005',  'Corn (Maize)',      'agriculture', 'ZC=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('1201',  'Soybeans',          'agriculture', 'ZS=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('1006',  'Rice',              'agriculture', 'ZR=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('1701',  'Sugar',             'soft',        'SB=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('0901',  'Coffee',            'soft',        'KC=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('1801',  'Cocoa',             'soft',        'CC=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('5201',  'Cotton',            'soft',        'CT=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('4403',  'Lumber (Wood)',     'soft',        'LBS=F', TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('0102',  'Live Cattle',       'livestock',   'LE=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('0103',  'Live Hogs',         'livestock',   'HE=F',  TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('8542',  'Semiconductors',    'tech',        'SMH',   TRUE) ON CONFLICT DO NOTHING;
INSERT INTO global.tracked_commodities VALUES ('8471',  'Computers',         'tech',        NULL,    TRUE) ON CONFLICT DO NOTHING;

-- ══════════════════════════════════════════
-- WATCHLIST HEALTH TRACKING
-- Accumulates data quality + analysis quality
-- signals per ticker across bot cycles.
-- Drives the auto-purge system.
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS ticker_health (
    ticker              TEXT PRIMARY KEY,
    -- Data quality signals (accumulated across cycles)
    total_cycles        INTEGER DEFAULT 0,
    news_article_count  INTEGER DEFAULT 0,
    reddit_post_count   INTEGER DEFAULT 0,
    youtube_count       INTEGER DEFAULT 0,
    zero_news_streak    INTEGER DEFAULT 0,
    collection_failures INTEGER DEFAULT 0,
    -- Analysis quality signals
    total_analyses      INTEGER DEFAULT 0,
    avg_confidence      DOUBLE PRECISION DEFAULT 0,
    hold_streak         INTEGER DEFAULT 0,
    last_action         TEXT,
    last_confidence     INTEGER DEFAULT 0,
    buy_count           INTEGER DEFAULT 0,
    sell_count          INTEGER DEFAULT 0,
    -- Computed score
    health_score        INTEGER DEFAULT 50,
    health_tier         TEXT DEFAULT 'new',
    -- Timestamps
    first_seen_at       TIMESTAMP,
    last_analyzed_at    TIMESTAMP,
    last_scored_at      TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extend watchlist with health + purge tracking
ALTER TABLE watchlist ADD COLUMN IF NOT EXISTS health_score INTEGER DEFAULT 50;
ALTER TABLE watchlist ADD COLUMN IF NOT EXISTS purged_at TIMESTAMP;
ALTER TABLE watchlist ADD COLUMN IF NOT EXISTS purge_reason TEXT;

-- ── 13F Hedge Fund Tracker tables ────────
CREATE TABLE IF NOT EXISTS sec_13f_filers (
    cik            TEXT PRIMARY KEY,
    filer_name     TEXT NOT NULL,
    last_checked   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active      BOOLEAN DEFAULT TRUE,
    latest_quarter TEXT,
    next_expected_filing DATE
);

-- ══════════════════════════════════════════
-- PIPELINE STATE
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS pipeline_state (
    singleton_id TEXT PRIMARY KEY,  -- Always 'current'
    status TEXT,
    cycle_id TEXT,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    tickers JSONB,
    progress TEXT,
    error TEXT,
    phase TEXT,
    operational_phase TEXT,
    step_count INTEGER,
    total_steps INTEGER,
    collect_flag BOOLEAN,
    analyze_flag BOOLEAN,
    trade_flag BOOLEAN
);

-- ══════════════════════════════════════════
-- PIPELINE EVENTS
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS pipeline_events (
    id TEXT PRIMARY KEY,
    cycle_id TEXT,
    timestamp TIMESTAMP,
    phase TEXT,
    step TEXT,
    detail TEXT,
    status TEXT,
    data_json JSONB,
    elapsed_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_pipeline_events_cycle ON pipeline_events(cycle_id);

-- ══════════════════════════════════════════
-- CYCLE CHECKPOINTS (resume after crash)
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS cycle_resume_state (
    cycle_id              TEXT PRIMARY KEY,
    status                TEXT DEFAULT 'interrupted',   -- 'interrupted' | 'expired'
    completed_phases      JSONB DEFAULT '[]',           -- ["collecting"]
    completed_tickers     JSONB DEFAULT '{}',           -- {"analyzing": ["NVDA","PLTR"]}
    cycle_config          JSONB DEFAULT '{}',           -- full cycle params snapshot
    checkpoint_ts         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    original_started_at   TIMESTAMP
);

-- ══════════════════════════════════════════
-- CYCLE RUN SUMMARIES
-- Canonical end-of-cycle diagnosis row.
-- Written in `finally` block — guaranteed even on crash.
-- One row per cycle. THE source of truth for:
--   - why there were no trades
--   - why there was no report
--   - whether Jetson was healthy
--   - what failed and why
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS cycle_run_summaries (
    cycle_id                TEXT PRIMARY KEY,
    trigger_type            TEXT DEFAULT 'manual',   -- 'manual' | 'scheduler'
    schedule_id             TEXT,                    -- FK to cycle_schedules.id (nullable)
    started_at              TIMESTAMP,
    finished_at             TIMESTAMP,
    status                  TEXT,                    -- 'done' | 'failed' | 'stopped' | 'error'
    elapsed_ms              INTEGER,
    -- Requested intent (what the user/scheduler asked for)
    tickers_requested       JSONB,                       -- original ticker list
    tickers_final           JSONB,                       -- after discovery/merge/filter
    collect_requested       BOOLEAN,
    analyze_requested       BOOLEAN,
    trade_requested         BOOLEAN,
    -- Jetson/vLLM health
    jetson_healthy_start    BOOLEAN,                    -- health check at cycle start
    -- Collection phase outcomes
    collector_ok            INTEGER DEFAULT 0,
    collector_skipped       INTEGER DEFAULT 0,
    collector_error         INTEGER DEFAULT 0,
    collector_failures      JSONB,                       -- ["reddit", "youtube"]
    -- Analysis phase outcomes
    analysis_results_count  INTEGER DEFAULT 0,
    buy_count               INTEGER DEFAULT 0,
    sell_count              INTEGER DEFAULT 0,
    hold_count              INTEGER DEFAULT 0,
    review_count            INTEGER DEFAULT 0,
    -- Trading phase outcomes
    trade_attempted         INTEGER DEFAULT 0,
    trade_executed          INTEGER DEFAULT 0,
    trade_failed            INTEGER DEFAULT 0,
    trade_skip_categories   JSONB,                       -- {"holds": N, "human_review": N, ...}
    -- Diagnosis
    no_trade_reason         TEXT,                    -- 'hold_only' | 'trading_disarmed' | 'no_analysis' | 'jetson_down' | 'zero_results' | null
    primary_failure_reason  TEXT,                    -- free text: first critical error
    -- Report (future-ready)
    report_generated        BOOLEAN DEFAULT FALSE,
    report_published        BOOLEAN DEFAULT FALSE,
    -- Catch-all
    summary_json            JSONB
);

CREATE INDEX IF NOT EXISTS idx_cycle_summaries_status ON cycle_run_summaries(status);
CREATE INDEX IF NOT EXISTS idx_cycle_summaries_trigger ON cycle_run_summaries(trigger_type);


-- ══════════════════════════════════════════
-- STRATEGY EVALUATIONS
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS strategy_evaluations (
    id              TEXT PRIMARY KEY,
    cycle_id        TEXT,
    timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_score     DOUBLE PRECISION,
    risk_score      DOUBLE PRECISION,
    performance_score DOUBLE PRECISION,
    robustness_score DOUBLE PRECISION,
    logic_score     DOUBLE PRECISION,
    operational_score DOUBLE PRECISION,
    full_analysis   JSONB
);

CREATE TABLE IF NOT EXISTS llm_attention_weights (
    id                  TEXT PRIMARY KEY,
    cycle_id            TEXT,
    agent_step          TEXT,
    node_id             TEXT,
    weight              DOUBLE PRECISION,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ══════════════════════════════════════════
-- ASI-EVOLVE STRATEGY EVOLUTION
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS evolution_nodes (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    round           INTEGER NOT NULL,
    parent_id       TEXT,
    motivation      TEXT,
    code            TEXT,
    metrics         TEXT,
    score           DOUBLE PRECISION,
    status          TEXT,
    analysis        TEXT,
    timestamp       TEXT
);

CREATE INDEX IF NOT EXISTS idx_evo_session ON evolution_nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_evo_score ON evolution_nodes(score);
CREATE INDEX IF NOT EXISTS idx_evo_status ON evolution_nodes(status);

CREATE TABLE IF NOT EXISTS evolution_lessons (
    id          TEXT PRIMARY KEY,
    session_id  TEXT,
    round       INTEGER,
    score       DOUBLE PRECISION,
    status      TEXT,
    lesson_text TEXT,
    timestamp   TEXT
);

CREATE INDEX IF NOT EXISTS idx_evo_lessons_session ON evolution_lessons(session_id);

-- ══════════════════════════════════════════
-- JIT SCRAPER QUEUE
-- Demand-driven data requests. Analysis agents
-- enqueue missing data here. scraper workers
-- consume by priority (1=JIT blocking, 5=routine).
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS scraper_queue (
    id                  TEXT PRIMARY KEY,
    ticker              TEXT NOT NULL,
    data_type_requested TEXT NOT NULL,    -- 'news' | 'reddit' | 'youtube' | 'price' | 'fundamentals' | 'options'
    priority            INTEGER DEFAULT 5,   -- 1=JIT (blocking analysis), 5=routine sweep
    status              TEXT DEFAULT 'PENDING',  -- PENDING | PROCESSING | RESOLVED | FAILED
    requested_by_lens   TEXT,             -- which lens requested this data
    retry_count         INTEGER DEFAULT 0,
    max_retries         INTEGER DEFAULT 3,
    error_message       TEXT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at          TIMESTAMP,
    resolved_at         TIMESTAMP,
    cooldown_until      TIMESTAMP            -- prevent infinite request loops
);
CREATE INDEX IF NOT EXISTS idx_scraper_queue_status ON scraper_queue(status, priority);
CREATE INDEX IF NOT EXISTS idx_scraper_queue_ticker ON scraper_queue(ticker);

-- ══════════════════════════════════════════
-- STRATEGY CANDIDATES
-- Multi-angle analysis outputs. Each lens pass
-- on a piece of data produces a candidate with
-- a signal and confidence score.
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS strategy_candidates (
    id                  TEXT PRIMARY KEY,
    cycle_id            TEXT,
    ticker              TEXT NOT NULL,
    lens_name           TEXT NOT NULL,    -- 'fundamental' | 'technical' | 'momentum' | 'risk' | custom
    system_prompt_hash  TEXT,             -- SHA256 of the system prompt used
    summary             TEXT,
    signal              TEXT,             -- BUY | SELL | HOLD
    confidence_score    INTEGER DEFAULT 0,
    analysis_result_id  TEXT,             -- FK to analysis_results.id
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_strategy_ticker ON strategy_candidates(ticker);
CREATE INDEX IF NOT EXISTS idx_strategy_lens ON strategy_candidates(lens_name);
CREATE INDEX IF NOT EXISTS idx_strategy_cycle ON strategy_candidates(cycle_id);

-- ══════════════════════════════════════════
-- STRATEGY PERFORMANCE
-- P&L tracking per system prompt. Links a
-- strategy candidate to its trade outcome so
-- we can rank prompts by win rate.
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS strategy_performance (
    id                      TEXT PRIMARY KEY,
    strategy_candidate_id   TEXT,             -- FK to strategy_candidates
    decision_outcome_id     TEXT,             -- FK to decision_outcomes
    agent_prompt_hash       TEXT,
    ticker                  TEXT,
    signal                  TEXT,             -- BUY | SELL | HOLD
    entry_price             DOUBLE PRECISION,
    exit_price              DOUBLE PRECISION,
    hold_days               INTEGER,
    return_pct              DOUBLE PRECISION,
    win                     BOOLEAN,
    active                  BOOLEAN DEFAULT TRUE,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at             TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_strat_perf_hash ON strategy_performance(agent_prompt_hash);
CREATE INDEX IF NOT EXISTS idx_strat_perf_ticker ON strategy_performance(ticker);
CREATE INDEX IF NOT EXISTS idx_strat_perf_active ON strategy_performance(active);

-- ══════════════════════════════════════════
-- META-AGENT GENERATED PROMPTS
-- LLM-generated analytical lenses. The meta-agent
-- reviews winning strategies and invents new system
-- prompts. Benched when win_rate drops below threshold.
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS generated_agent_prompts (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    lens_type           TEXT DEFAULT 'custom',  -- analytical lens category
    system_prompt       TEXT NOT NULL,
    prompt_hash         TEXT NOT NULL,           -- SHA256 for dedup
    performance_score   DOUBLE PRECISION DEFAULT 0.0,
    total_trades        INTEGER DEFAULT 0,
    win_rate            DOUBLE PRECISION DEFAULT 0.0,
    active              BOOLEAN DEFAULT TRUE,
    created_by          TEXT DEFAULT 'meta_agent',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at        TIMESTAMP,
    benched_at          TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_gen_prompts_active ON generated_agent_prompts(active);
CREATE INDEX IF NOT EXISTS idx_gen_prompts_hash ON generated_agent_prompts(prompt_hash);

-- ══════════════════════════════════════════
-- SWARM SCORECARDS (model accuracy tracking)
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS swarm_scorecards (
    id                      TEXT PRIMARY KEY,
    ticker                  TEXT NOT NULL,
    cycle_id                TEXT NOT NULL,
    model_label             TEXT NOT NULL,    -- quant_26B | macro_35B | cio_120B
    model_id                TEXT,             -- actual model name from vLLM
    predicted_action        TEXT,             -- BUY/SELL/HOLD
    predicted_confidence    INTEGER,
    predicted_price_target  DOUBLE PRECISION,
    predicted_stop_loss     DOUBLE PRECISION,
    key_signals             TEXT,             -- JSON array of signals
    rationale               TEXT,
    actual_action           TEXT,             -- Filled by grading pass
    actual_price_change_pct DOUBLE PRECISION, -- Filled by grading pass
    accuracy_score          DOUBLE PRECISION, -- 0-100, filled by grading
    action_correct          BOOLEAN,          -- Filled by grading
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    graded_at               TIMESTAMP         -- NULL until graded
);

CREATE INDEX IF NOT EXISTS idx_scorecard_ticker ON swarm_scorecards(ticker);
CREATE INDEX IF NOT EXISTS idx_scorecard_cycle ON swarm_scorecards(cycle_id);
CREATE INDEX IF NOT EXISTS idx_scorecard_model ON swarm_scorecards(model_id);

-- ══════════════════════════════════════════
-- PERFORMANCE INDEXES (PostgreSQL-specific)
-- ══════════════════════════════════════════

-- BRIN indexes for time-series data (skip-scan over time blocks)
CREATE INDEX IF NOT EXISTS idx_price_history_date_brin ON price_history USING brin(date);
CREATE INDEX IF NOT EXISTS idx_pipeline_events_ts_brin ON pipeline_events USING brin(timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_audit_created_brin ON llm_audit_logs USING brin(created_at);

-- HNSW vector indexes (pgvector — fast ANN search)
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_ontology_nodes_hnsw ON ontology_nodes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_user_data_hnsw ON user_data USING hnsw (embedding vector_cosine_ops);

-- GIN indexes on JSONB columns
CREATE INDEX IF NOT EXISTS idx_pipeline_state_tickers_gin ON pipeline_state USING gin(tickers);
CREATE INDEX IF NOT EXISTS idx_cycle_summaries_json_gin ON cycle_run_summaries USING gin(summary_json);

-- ══════════════════════════════════════════
-- AUTORESEARCH REPORTS
-- Post-cycle holistic audit + LLM reflection
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS autoresearch_reports (
    id                      TEXT PRIMARY KEY,
    cycle_id                TEXT NOT NULL,
    created_at              TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Summary scores (0-100)
    data_quality_score      DOUBLE PRECISION,
    decision_quality_score  DOUBLE PRECISION,
    llm_performance_score   DOUBLE PRECISION,
    overall_score           DOUBLE PRECISION,

    -- Detailed findings (JSON blobs)
    data_gaps               TEXT,       -- JSON: [{ticker, missing_sources, recommendation}]
    decision_issues         TEXT,       -- JSON: [{ticker, action, issue, suggestion}]
    llm_issues              TEXT,       -- JSON: [{model, agent, issue, count}]
    performance_metrics     TEXT,       -- JSON: {total_ms, cache_hit_pct, ...}

    -- LLM Reflection output
    reflection              TEXT,       -- JSON: {summary, recommendations[], adjustments[]}

    -- Recovery/resilience data absorbed here
    recovery_stats          TEXT,       -- JSON: {total_failures, by_type, circuit_breakers}

    -- Status
    status                  TEXT DEFAULT 'running',  -- running | done | error
    phase                   TEXT DEFAULT '',
    error                   TEXT
);

CREATE INDEX IF NOT EXISTS idx_autoresearch_cycle ON autoresearch_reports(cycle_id);
CREATE INDEX IF NOT EXISTS idx_autoresearch_created ON autoresearch_reports(created_at);

-- ══════════════════════════════════════════
-- HALLUCINATION AUDIT LOG
-- Post-LLM verification results; tracks rejected
-- and flagged hallucinations for training data
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS hallucination_log (
    id                  TEXT PRIMARY KEY,
    ticker              TEXT,
    cycle_id            TEXT,
    hallucination_count INTEGER,
    total_claims        INTEGER,
    hallucination_rate  DOUBLE PRECISION,
    rejected            BOOLEAN,
    details_json        TEXT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hallucination_audit (
    id                  TEXT PRIMARY KEY,
    cycle_id            TEXT,
    ticker              TEXT,
    source_file         TEXT,
    foreign_value       TEXT,
    context_snippet     TEXT,
    created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_hallucination_ticker ON hallucination_log(ticker);
CREATE INDEX IF NOT EXISTS idx_hallucination_rejected ON hallucination_log(rejected);

-- ══════════════════════════════════════════
-- PENDING EVOLUTION FIXES
-- System-level fixes (prompts, scrapers) generated by
-- the Evolution Debate Council, awaiting user approval
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS pending_evolution_fixes (
    id                      TEXT PRIMARY KEY,
    cycle_id                TEXT NOT NULL,
    target_type             TEXT NOT NULL,    -- 'prompt', 'scraper', 'strategy'
    target_name             TEXT NOT NULL,    -- e.g. 'debate_prompts.py', 'reddit_scraper'
    proposed_fix            TEXT NOT NULL,    -- JSON or raw text of the fix
    motivation              TEXT NOT NULL,
    proposer_model          TEXT,
    critic_concerns         TEXT,
    judge_score             DOUBLE PRECISION,
    status                  TEXT DEFAULT 'pending', -- pending | approved | rejected | deployed | rolled_back
    backup_path             TEXT,             -- path to pre-deploy backup file
    probation_until         TIMESTAMP,        -- auto-rollback monitoring window
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at             TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pending_evo_fixes_status ON pending_evolution_fixes(status);

-- ══════════════════════════════════════════
-- STABLE HARNESS REGISTRY
-- Tracks the last known-good version of evolved files.
-- Written by mark_stable() after probation passes.
-- Read by the Debate Council as a fallback starting point.
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS stable_harnesses (
    target_type     TEXT NOT NULL,
    target_name     TEXT NOT NULL,
    fix_id          TEXT NOT NULL,         -- FK to pending_evolution_fixes.id
    stable_content  TEXT NOT NULL,         -- snapshot of the known-good file content
    marked_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (target_type, target_name)
);

-- ══════════════════════════════════════════
-- SANDBOX COMMAND APPROVALS
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS pending_approvals (
    id              TEXT PRIMARY KEY,
    agent_name      TEXT,
    command         TEXT,
    reason          TEXT,
    status          TEXT DEFAULT 'pending', -- pending, approved, rejected
    stdout          TEXT,
    stderr          TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at     TIMESTAMP
);

-- ══════════════════════════════════════════
-- LENS SCORECARD
-- Performance tracking for individual prompts
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS lens_scorecard (
    id                      TEXT PRIMARY KEY,
    lens_name               TEXT NOT NULL,
    lens_type               TEXT NOT NULL,
    system_prompt           TEXT,
    cycle_id                TEXT NOT NULL,
    ticker                  TEXT NOT NULL,
    predicted_action        TEXT,
    predicted_confidence    INTEGER,
    actual_action           TEXT,
    actual_price_change_pct DOUBLE PRECISION,
    accuracy_score          DOUBLE PRECISION,
    action_correct          BOOLEAN,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    graded_at               TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lens_scorecard_ticker ON lens_scorecard(ticker);
CREATE INDEX IF NOT EXISTS idx_lens_scorecard_lens ON lens_scorecard(lens_name);

-- ══════════════════════════════════════════
-- AGENT LOOP STATS
-- Tracking multi-turn agentic loop governance
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS agent_loop_stats (
    id                      TEXT PRIMARY KEY,
    cycle_id                TEXT,
    agent_name              TEXT,
    ticker                  TEXT,
    loops_used              INTEGER,
    token_usage             INTEGER,
    cost_usd                DOUBLE PRECISION,
    yielded                 BOOLEAN,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ══════════════════════════════════════════
-- EVOLUTION DEAD ENDS
-- Rolled-back fixes that failed — prevents
-- the debate council from repeating mistakes
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS evolution_dead_ends (
    id              TEXT PRIMARY KEY,
    fix_id          TEXT NOT NULL,
    target_type     TEXT NOT NULL,
    target_name     TEXT NOT NULL,
    approach_hash   TEXT NOT NULL,
    failure_reason  TEXT NOT NULL,
    metrics_before  JSONB,
    metrics_after   JSONB,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dead_ends_target
    ON evolution_dead_ends(target_type, target_name);

-- ══════════════════════════════════════════
-- SUBSYSTEM BENCHMARKS
-- Per-cycle metrics for every pipeline subsystem
-- Powers trend analysis and rollback decisions
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS subsystem_benchmarks (
    id              TEXT PRIMARY KEY,
    cycle_id        TEXT NOT NULL,
    subsystem       TEXT NOT NULL,
    metrics         JSONB NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sub_bench_cycle ON subsystem_benchmarks(cycle_id);
CREATE INDEX IF NOT EXISTS idx_sub_bench_subsystem ON subsystem_benchmarks(subsystem);

-- ══════════════════════════════════════════
-- TOOL USAGE STATS
-- Per-invocation metrics for every tool call.
-- Powers the frontend Tools dashboard and
-- enables the bot to self-diagnose tool neglect.
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS tool_usage_stats (
    id              SERIAL PRIMARY KEY,
    tool_name       TEXT NOT NULL,
    agent_name      TEXT DEFAULT '',
    ticker          TEXT DEFAULT '',
    cycle_id        TEXT DEFAULT '',
    success         BOOLEAN DEFAULT TRUE,
    execution_ms    INTEGER DEFAULT 0,
    error_message   TEXT,
    called_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tool_usage_name ON tool_usage_stats(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_usage_ts ON tool_usage_stats(called_at);

-- ══════════════════════════════════════════
-- CYCLE AUDIT LOG (forensic diagnostics)
-- Structured diagnostic events emitted by
-- cycle_auditor.py at critical pipeline
-- transition points.
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS cycle_audit_log (
    id          TEXT PRIMARY KEY,
    cycle_id    TEXT NOT NULL,
    timestamp   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    audit_type  TEXT NOT NULL,       -- phase_entry | phase_exit | ticker_result | llm_response | anomaly
    phase       TEXT DEFAULT '',
    ticker      TEXT DEFAULT '',
    severity    TEXT DEFAULT 'info', -- info | warning | critical
    message     TEXT,
    data        JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_log_cycle ON cycle_audit_log(cycle_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_severity ON cycle_audit_log(severity);
CREATE INDEX IF NOT EXISTS idx_audit_log_type ON cycle_audit_log(audit_type);

-- ══════════════════════════════════════════
-- DEBATE HISTORY (warm-start memory)
-- Records debate outcomes per ticker per cycle
-- so cycle_brief_builder can inject prior
-- debate winners into the LLM context.
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS debate_history (
    id                  TEXT PRIMARY KEY,
    cycle_id            TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    thesis_action       TEXT,           -- BUY/SELL/HOLD (Config C)
    thesis_confidence   INTEGER,
    counter_action      TEXT,           -- Devil's advocate position
    counter_confidence  INTEGER,
    winner              TEXT,           -- 'thesis' or 'antithesis'
    final_action        TEXT,           -- Synthesis result
    final_confidence    INTEGER,
    persona_name        TEXT,
    key_risk            TEXT,
    pro_argument        TEXT,
    con_argument        TEXT,
    persona_outcomes    JSONB,
    created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_debate_history_ticker ON debate_history(ticker);
CREATE INDEX IF NOT EXISTS idx_debate_history_cycle ON debate_history(cycle_id);

-- ══════════════════════════════════════════
-- CYCLE SUMMARIES (warm-start memory)
-- LLM-facing distilled end-of-cycle stats.
-- Distinct from cycle_run_summaries which is
-- the raw orchestrator diagnostic row.
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS cycle_summaries (
    id              TEXT PRIMARY KEY,
    cycle_id        TEXT NOT NULL UNIQUE,
    total_tickers   INTEGER,
    buy_count       INTEGER DEFAULT 0,
    sell_count      INTEGER DEFAULT 0,
    hold_count      INTEGER DEFAULT 0,
    avg_confidence  DOUBLE PRECISION,
    top_ticker      TEXT,
    top_confidence  INTEGER,
    lesson_summary  TEXT,
    ticker          TEXT,                  -- per-ticker summaries (nullable for aggregate rows)
    action          TEXT,                  -- BUY/SELL/HOLD decision for this ticker
    cycle_date      TIMESTAMPTZ,          -- when this cycle ran
    was_correct     BOOLEAN,              -- backfilled from decision_outcomes: TRUE=WIN, FALSE=LOSS, NULL=unresolved
    outcome_pnl     DOUBLE PRECISION,     -- backfilled: realized pnl_pct from decision_outcomes
    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cycle_summaries_cycle ON cycle_summaries(cycle_id);

-- ══════════════════════════════════════════
-- BEST PER TICKER (warm-start memory baseline)
-- Stores the historical best performance per ticker.
-- Replaces JSON fixtures.
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS best_per_ticker (
    ticker          TEXT PRIMARY KEY,
    action          TEXT,
    confidence      INTEGER,
    rationale       TEXT,
    is_correct      BOOLEAN,
    score           DOUBLE PRECISION,
    updated_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ══════════════════════════════════════════
-- TOOL-USE IMPROVEMENT FRAMEWORK
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS agent_traces (
    id                  TEXT PRIMARY KEY,
    run_id              TEXT,
    agent_name          TEXT,
    task_type           TEXT,
    goal                TEXT,
    planned_next_action TEXT,
    tool_name           TEXT,
    tool_args           TEXT,
    tool_result_summary TEXT,
    why_tool_was_called TEXT,
    tokens_before       INTEGER,
    tokens_after        INTEGER,
    latency_ms          INTEGER,
    did_tool_change_decision BOOLEAN,
    loop_step           INTEGER,
    stop_reason         TEXT,
    endpoint_name       TEXT,
    model_name          TEXT,
    created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_agent_traces_run ON agent_traces(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_traces_agent ON agent_traces(agent_name);

CREATE TABLE IF NOT EXISTS eval_scores (
    id                  TEXT PRIMARY KEY,
    run_id              TEXT,
    completion_score    DOUBLE PRECISION,
    tool_correctness_score DOUBLE PRECISION,
    efficiency_score    DOUBLE PRECISION,
    error_recovery_score DOUBLE PRECISION,
    stop_quality_score  DOUBLE PRECISION,
    final_score         DOUBLE PRECISION,
    created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_eval_scores_run ON eval_scores(run_id);

CREATE TABLE IF NOT EXISTS failure_buckets (
    id                  TEXT PRIMARY KEY,
    run_id              TEXT,
    bucket_type         TEXT, -- skipped_needed_tool, wrong_tool_selected, etc.
    description         TEXT,
    created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_failure_buckets_run ON failure_buckets(run_id);

CREATE TABLE IF NOT EXISTS tool_playbook (
    id                  TEXT PRIMARY KEY,
    task_type           TEXT,
    market_context      TEXT,
    agent_role          TEXT,
    recommended_tool_sequence TEXT,
    required_preconditions TEXT,
    stop_conditions     TEXT,
    bad_patterns_to_avoid TEXT,
    example_good_trace_id TEXT,
    score_stats         TEXT,
    last_validated_at   TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);


-- TICKER CONSENSUS
-- Stores market consensus per ticker.
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS ticker_consensus (
    ticker          TEXT PRIMARY KEY,
    consensus       TEXT,
    last_updated    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
