"""
Database Migrations
Handles structural updates to the schema for existing PostgreSQL databases.
Runs lightweight auto-migrations to ensure compatibility with newly added columns.
"""


def _safe_add_column(conn, table: str, column: str, dtype: str):
    """Add a column if it doesn't exist.

    PostgreSQL supports ADD COLUMN IF NOT EXISTS natively (since v9.6).
    We use a cursor from the raw connection (not PooledCursor) since
    this is called during pool init.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {dtype}"
            )
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass


def run_migrations(conn):
    """Auto-migrations for existing databases to match the current schema_pg.sql."""
    # ── Layout Presets (cross-browser sync)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS layout_presets (
                    name        TEXT PRIMARY KEY,
                    layout_data JSONB NOT NULL,
                    is_active   BOOLEAN DEFAULT FALSE,
                    updated_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Youtube
    _safe_add_column(conn, "youtube_transcripts", "thumbnail_url", "TEXT")
    _safe_add_column(conn, "youtube_transcripts", "summary", "TEXT")
    _safe_add_column(conn, "youtube_transcripts", "tickers_mentioned", "TEXT")
    _safe_add_column(conn, "youtube_transcripts", "summarized_at", "TIMESTAMPTZ")

    # ── Timestamps & Quality
    _safe_add_column(conn, "news_articles", "collected_at", "TIMESTAMPTZ")
    _safe_add_column(conn, "news_articles", "quality_status", "TEXT")
    _safe_add_column(conn, "news_articles", "quality_reason", "TEXT")
    _safe_add_column(conn, "news_articles", "quality_score", "INTEGER")
    _safe_add_column(conn, "reddit_posts", "collected_at", "TIMESTAMPTZ")
    _safe_add_column(conn, "reddit_posts", "quality_status", "TEXT")
    _safe_add_column(conn, "reddit_posts", "quality_reason", "TEXT")
    _safe_add_column(conn, "youtube_transcripts", "collected_at", "TIMESTAMPTZ")
    _safe_add_column(conn, "youtube_transcripts", "quality_status", "TEXT")
    _safe_add_column(conn, "youtube_transcripts", "quality_reason", "TEXT")

    # ── URL Dedup constraint removed intentionally ──
    # The UNIQUE(url) constraint was removed because multiple tickers can share the same article URL,
    # and the primary key id (hash of title+ticker) handles deduplication correctly.

    # ── Source Trust (Reputation system updates)
    _safe_add_column(conn, "source_trust", "win_rate", "DOUBLE PRECISION DEFAULT 0.0")
    _safe_add_column(conn, "source_trust", "quality_wins", "INTEGER DEFAULT 0")
    _safe_add_column(conn, "source_trust", "flag_rate", "DOUBLE PRECISION DEFAULT 0.0")

    # ── Quant-grade commodity correlation columns
    _safe_add_column(
        conn, "stock_commodity_correlations", "cointegration_pvalue", "DOUBLE PRECISION"
    )
    _safe_add_column(conn, "stock_commodity_correlations", "cointegrated", "BOOLEAN")
    _safe_add_column(conn, "stock_commodity_correlations", "lead_lag_days", "INTEGER")
    _safe_add_column(
        conn, "stock_commodity_correlations", "lead_lag_correlation", "DOUBLE PRECISION"
    )
    _safe_add_column(
        conn, "stock_commodity_correlations", "vol_adj_correlation", "DOUBLE PRECISION"
    )
    _safe_add_column(
        conn,
        "stock_commodity_correlations",
        "correlation_stability",
        "DOUBLE PRECISION",
    )
    _safe_add_column(
        conn, "stock_commodity_correlations", "distance_correlation", "DOUBLE PRECISION"
    )
    _safe_add_column(conn, "stock_commodity_correlations", "quant_score", "INTEGER")
    _safe_add_column(conn, "stock_commodity_correlations", "method_details", "TEXT")

    # ── 13F Hedge Fund Tracker
    _safe_add_column(conn, "sec_13f_filers", "latest_quarter", "TEXT")
    _safe_add_column(conn, "sec_13f_filers", "next_expected_filing", "DATE")
    _safe_add_column(conn, "sec_13f_holdings", "name_of_issuer", "TEXT")
    _safe_add_column(conn, "sec_13f_holdings", "cusip", "TEXT")
    _safe_add_column(conn, "sec_13f_holdings", "share_type", "TEXT")
    _safe_add_column(conn, "sec_13f_holdings", "pct_change", "DOUBLE PRECISION")
    _safe_add_column(conn, "sec_13f_holdings", "is_new_position", "BOOLEAN")
    _safe_add_column(conn, "sec_13f_holdings", "is_exit", "BOOLEAN")
    _safe_add_column(conn, "sec_13f_holdings", "filing_date", "DATE")
    _safe_add_column(conn, "sec_13f_holdings", "collected_at", "TIMESTAMPTZ")

    # ── Scheduler: max_tickers per scheduled run
    _safe_add_column(conn, "cycle_schedules", "max_tickers", "INTEGER")

    # ── Pipeline version routing / benchmarking
    _safe_add_column(conn, "cycle_benchmarks", "requested_version", "TEXT")
    _safe_add_column(conn, "cycle_benchmarks", "effective_version", "TEXT")
    _safe_add_column(conn, "cycle_benchmarks", "benchmark_group", "TEXT")
    _safe_add_column(conn, "cycle_benchmarks", "execution_mode", "TEXT")
    _safe_add_column(conn, "cycle_benchmarks", "v2_stage", "INTEGER")
    _safe_add_column(conn, "pipeline_state", "requested_pipeline_version", "TEXT")
    _safe_add_column(conn, "pipeline_state", "effective_pipeline_version", "TEXT")
    _safe_add_column(conn, "pipeline_state", "benchmark_group", "TEXT")
    _safe_add_column(conn, "pipeline_state", "execution_mode", "TEXT")
    _safe_add_column(conn, "pipeline_state", "v2_stage", "INTEGER")

    # ── Strategy Evaluations: persist scope cycle_id
    _safe_add_column(conn, "strategy_evaluations", "cycle_id", "TEXT")

    # ── Ontology Graph: source_cycle_id tracking
    _safe_add_column(conn, "ontology_nodes", "source_cycle_id", "TEXT")

    # ── JIT Scraper / Re-analysis tracking
    _safe_add_column(conn, "news_articles", "analysis_count", "INTEGER DEFAULT 0")
    _safe_add_column(conn, "news_articles", "max_analyses", "INTEGER DEFAULT 5")
    _safe_add_column(conn, "reddit_posts", "analysis_count", "INTEGER DEFAULT 0")
    _safe_add_column(conn, "reddit_posts", "max_analyses", "INTEGER DEFAULT 5")
    _safe_add_column(conn, "youtube_transcripts", "analysis_count", "INTEGER DEFAULT 0")
    _safe_add_column(conn, "youtube_transcripts", "max_analyses", "INTEGER DEFAULT 5")

    # ── Attention Tracker (Smart Ticker Triage) ──
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ticker_attention (
                    ticker              TEXT PRIMARY KEY,
                    last_collected_at   TIMESTAMPTZ,
                    last_analyzed_at    TIMESTAMPTZ,
                    last_traded_at      TIMESTAMPTZ,
                    consecutive_skips   INTEGER DEFAULT 0,
                    consecutive_holds   INTEGER DEFAULT 0,
                    days_since_deep     INTEGER DEFAULT 0,
                    neglect_flagged     BOOLEAN DEFAULT FALSE,
                    neglect_reason      TEXT,
                    data_hash           TEXT,
                    created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Cycle Directives (Autoresearch Self-Improvement) ──
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cycle_directives (
                    id               TEXT PRIMARY KEY,
                    cycle_id         TEXT NOT NULL,
                    directive_type   TEXT NOT NULL,
                    directive_text   TEXT NOT NULL,
                    target_ticker    TEXT,
                    severity         TEXT DEFAULT 'info',
                    status           TEXT DEFAULT 'active',
                    created_at       TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    resolved_at      TIMESTAMPTZ,
                    expires_after    INTEGER DEFAULT 5
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_directives_status "
                "ON cycle_directives(status)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_directives_cycle "
                "ON cycle_directives(cycle_id)"
            )
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Triage tier audit column on analysis_results ──
    _safe_add_column(conn, "analysis_results", "triage_tier", "TEXT")

    # ── Maintenance agent retry tracking ──
    _safe_add_column(
        conn, "pending_evolution_fixes", "attempt_count", "INTEGER DEFAULT 0"
    )

    # ── Rollback safety columns on pending_evolution_fixes ──
    _safe_add_column(conn, "pending_evolution_fixes", "backup_path", "TEXT")
    _safe_add_column(conn, "pending_evolution_fixes", "probation_until", "TIMESTAMPTZ")

    # ── Evolution Dead Ends (prevent repeating failed approaches) ──
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evolution_dead_ends (
                    id              TEXT PRIMARY KEY,
                    fix_id          TEXT NOT NULL,
                    target_type     TEXT NOT NULL,
                    target_name     TEXT NOT NULL,
                    approach_hash   TEXT NOT NULL,
                    failure_reason  TEXT NOT NULL,
                    metrics_before  JSONB,
                    metrics_after   JSONB,
                    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_dead_ends_target "
                "ON evolution_dead_ends(target_type, target_name)"
            )
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Pending Approvals ──
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pending_approvals (
                    id            TEXT PRIMARY KEY,
                    agent_name    TEXT,
                    command       TEXT,
                    reason        TEXT,
                    status        TEXT DEFAULT 'pending',
                    stdout        TEXT,
                    stderr        TEXT,
                    created_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    resolved_at   TIMESTAMPTZ
                )
            """)
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Subsystem Benchmarks (per-cycle per-subsystem metrics) ──
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS subsystem_benchmarks (
                    id              TEXT PRIMARY KEY,
                    cycle_id        TEXT NOT NULL,
                    subsystem       TEXT NOT NULL,
                    metrics         JSONB NOT NULL,
                    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_sub_bench_cycle "
                "ON subsystem_benchmarks(cycle_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_sub_bench_subsystem "
                "ON subsystem_benchmarks(subsystem)"
            )
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── One-time fix: ETH price collision + CAGR garbage position ──
    # The ticker "ETH" was misclassified as crypto, causing snapshots to use
    # the Ethereum crypto price (~$1,800) instead of the ETF price (~$23).
    # This inflated portfolio_snapshots to $324K.  CAGR had 479M shares at
    # $0.00001 due to a bad price pull.
    # This migration is idempotent: it only deletes if the bad data exists.
    _fix_eth_cagr_data(conn)


def _fix_eth_cagr_data(conn):
    """One-time fix for corrupted portfolio data."""
    import logging

    logger = logging.getLogger(__name__)
    BOT_ID = "lazy-trader-v4"

    try:
        with conn.cursor() as cur:
            # 1. Delete inflated snapshots (ETH price collision caused $324K)
            cur.execute("DELETE FROM portfolio_snapshots WHERE total_value > 200000")
            deleted_snaps = cur.rowcount
            if deleted_snaps > 0:
                logger.info(
                    "[MIGRATION] Deleted %d corrupted snapshot(s) with inflated values",
                    deleted_snaps,
                )

            # 2. Delete CAGR garbage position (479M shares @ $0.00001)
            cur.execute(
                "SELECT qty, avg_entry_price FROM positions "
                "WHERE bot_id = %s AND ticker = 'CAGR'",
                (BOT_ID,),
            )
            cagr = cur.fetchone()
            if cagr:
                cost = float(cagr[0]) * float(cagr[1])
                cur.execute(
                    "DELETE FROM positions WHERE bot_id = %s AND ticker = 'CAGR'",
                    (BOT_ID,),
                )
                cur.execute(
                    "UPDATE bots SET cash_balance = cash_balance + %s WHERE bot_id = %s",
                    (cost, BOT_ID),
                )
                cur.execute(
                    "DELETE FROM position_lots WHERE bot_id = %s AND ticker = 'CAGR'",
                    (BOT_ID,),
                )
                cur.execute(
                    "DELETE FROM orders WHERE bot_id = %s AND ticker = 'CAGR'",
                    (BOT_ID,),
                )
                cur.execute(
                    "DELETE FROM trade_fills WHERE bot_id = %s AND ticker = 'CAGR'",
                    (BOT_ID,),
                )
                logger.info(
                    "[MIGRATION] Deleted CAGR garbage position (%.0f shares), "
                    "refunded $%.2f to cash",
                    cagr[0],
                    cost,
                )

            conn.commit()
    except Exception as e:
        logger.warning("[MIGRATION] ETH/CAGR data fix failed (non-fatal): %s", e)
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Per-Box Telemetry columns on llm_audit_logs ──
    _safe_add_column(conn, "llm_audit_logs", "endpoint_name", "TEXT")
    _safe_add_column(conn, "llm_audit_logs", "prompt_tokens", "INTEGER DEFAULT 0")
    _safe_add_column(conn, "llm_audit_logs", "completion_tokens", "INTEGER DEFAULT 0")
    _safe_add_column(conn, "llm_audit_logs", "queue_wait_ms", "INTEGER DEFAULT 0")
    _safe_add_column(conn, "llm_audit_logs", "tokens_per_second", "REAL")

    # ── Triage tier on analysis_results ──
    _safe_add_column(conn, "analysis_results", "triage_tier", "TEXT DEFAULT 'standard'")



    # ── Evolution lessons archive (memory consolidation) ──
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evolution_lessons_archive (
                    id          TEXT PRIMARY KEY,
                    session_id  TEXT,
                    round       INTEGER DEFAULT 0,
                    score       REAL,
                    status      TEXT,
                    lesson_text TEXT,
                    timestamp   TEXT,
                    archived_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Fix discovered_tickers PK: (ticker) → (ticker, source) ──
    # The ON CONFLICT (ticker, source) clause in reddit_collector requires a
    # composite unique constraint that the old schema (PK on ticker only) doesn't have.
    try:
        with conn.cursor() as cur:
            # Check if 'source' is already part of the PK by seeing if the
            # unique index on (ticker, source) already exists
            cur.execute("""
                SELECT 1 FROM pg_indexes
                WHERE tablename = 'discovered_tickers'
                AND indexdef LIKE '%%ticker, source%%'
            """)
            if not cur.fetchone():
                cur.execute("""
                    ALTER TABLE discovered_tickers
                    DROP CONSTRAINT IF EXISTS
                    discovered_tickers_pkey
                """)
                cur.execute("""
                    ALTER TABLE discovered_tickers
                    ADD CONSTRAINT discovered_tickers_pkey
                    PRIMARY KEY (ticker, source)
                """)
                conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Hallucination Log (post-LLM verification audit trail) ──
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS hallucination_log (
                    id VARCHAR PRIMARY KEY,
                    ticker VARCHAR,
                    cycle_id VARCHAR,
                    hallucination_count INTEGER,
                    total_claims INTEGER,
                    hallucination_rate FLOAT,
                    rejected BOOLEAN,
                    details_json VARCHAR,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass



    # ── Market Snapshots (Anti-Hallucination V2) ──
    try:
        with conn.cursor() as cur:
            cur.execute("""
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
                )
            """)
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Debate History Unique Constraint and Columns ──
    _safe_add_column(conn, "debate_history", "pro_argument", "TEXT")
    _safe_add_column(conn, "debate_history", "con_argument", "TEXT")
    _safe_add_column(conn, "debate_history", "persona_outcomes", "JSONB")
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM debate_history a USING (
                  SELECT MIN(ctid) as ctid, ticker, cycle_id
                  FROM debate_history
                  GROUP BY ticker, cycle_id HAVING COUNT(*) > 1
                ) b
                WHERE a.ticker = b.ticker AND a.cycle_id = b.cycle_id AND a.ctid <> b.ctid
            """)
            cur.execute("""
                ALTER TABLE debate_history 
                ADD CONSTRAINT debate_history_ticker_cycle_id_key UNIQUE (ticker, cycle_id)
            """)
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Bot Profiles: starting_cash + description columns ──
    _safe_add_column(conn, "bots", "starting_cash", "DOUBLE PRECISION DEFAULT 100000.0")
    _safe_add_column(conn, "bots", "description", "TEXT DEFAULT ''")

    # Backfill starting_cash for existing bots that have NULL
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE bots SET starting_cash = cash_balance "
                "WHERE starting_cash IS NULL OR starting_cash = 0"
            )
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Trading Constitution (self-improving agentic rules) ──
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trading_constitution (
                    id              TEXT PRIMARY KEY,
                    rule_category   TEXT NOT NULL,
                    rule_text       TEXT NOT NULL,
                    rule_params     JSONB DEFAULT '{}',
                    version         INTEGER DEFAULT 1,
                    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    amended_at      TIMESTAMPTZ,
                    amendment_reason TEXT,
                    performance_data JSONB DEFAULT '{}',
                    is_active       BOOLEAN DEFAULT TRUE
                )
            """)
            # Seed v1 rules if table is empty
            cur.execute("SELECT COUNT(*) FROM trading_constitution")
            count = cur.fetchone()[0]
            if count == 0:
                import json as _json

                seed_rules = [
                    (
                        "position_limit_v1",
                        "position_limits",
                        "Maximum 8 concurrent open positions",
                        _json.dumps({"max_positions": 8}),
                    ),
                    (
                        "sector_concentration_v1",
                        "sector",
                        ("No more than 30% of positions in a single sector"),
                        _json.dumps({"max_sector_pct": 30}),
                    ),
                    (
                        "sell_rsi_v1",
                        "sell_triggers",
                        ("SELL if RSI > 70 (overbought condition)"),
                        _json.dumps({"rsi_threshold": 70}),
                    ),
                    (
                        "sell_pe_v1",
                        "sell_triggers",
                        ("SELL if P/E exceeds 1.5x sector average"),
                        _json.dumps({"pe_multiplier": 1.5}),
                    ),
                    (
                        "sell_holding_v1",
                        "sell_triggers",
                        ("Review positions held >14 days without thesis confirmation"),
                        _json.dumps(
                            {
                                "max_holding_days": 14,
                            }
                        ),
                    ),
                    (
                        "sizing_v1",
                        "sizing",
                        ("Position size 2-10% of cash based on confidence level"),
                        _json.dumps(
                            {
                                "min_pct": 2,
                                "max_pct": 10,
                                "min_confidence": 70,
                            }
                        ),
                    ),
                    (
                        "buy_rsi_v1",
                        "buy_requirements",
                        "BUY only if RSI < 65 (not overbought)",
                        _json.dumps({"rsi_max": 65}),
                    ),
                ]
                for rule_id, cat, text, params in seed_rules:
                    cur.execute(
                        "INSERT INTO trading_constitution "
                        "(id, rule_category, rule_text, "
                        "rule_params) "
                        "VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT (id) DO NOTHING",
                        (rule_id, cat, text, params),
                    )
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
    # ── Price Triggers: expanded columns for order trigger system ──
    _safe_add_column(conn, "price_triggers", "action", "TEXT DEFAULT 'SELL'")
    _safe_add_column(conn, "price_triggers", "qty_pct", "DOUBLE PRECISION DEFAULT 1.0")
    _safe_add_column(conn, "price_triggers", "trailing_pct", "DOUBLE PRECISION")
    _safe_add_column(conn, "price_triggers", "highest_price", "DOUBLE PRECISION")
    _safe_add_column(conn, "price_triggers", "reason", "TEXT")
    _safe_add_column(conn, "price_triggers", "triggered_at", "TIMESTAMPTZ")
    _safe_add_column(conn, "price_triggers", "created_by", "TEXT DEFAULT 'bot'")
    _safe_add_column(conn, "price_triggers", "trigger_price", "DOUBLE PRECISION")

    # ── Autoresearch Experiences (Reflector Loop) ──
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_experiences (
                    id              TEXT PRIMARY KEY,
                    agent_name      TEXT NOT NULL,
                    task_context    TEXT NOT NULL,
                    lesson_learned  TEXT NOT NULL,
                    success_score   DOUBLE PRECISION DEFAULT 1.0,
                    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    last_applied    TIMESTAMP WITH TIME ZONE
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_experiences_name "
                "ON agent_experiences(agent_name)"
            )
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Tool-Use Improvement Framework ──
    _safe_add_column(conn, "agent_traces", "task_type", "TEXT")
    _safe_add_column(conn, "agent_traces", "endpoint_name", "TEXT")
    _safe_add_column(conn, "agent_traces", "model_name", "TEXT")
    _safe_add_column(conn, "tool_playbook", "task_type", "TEXT")
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tool_playbook_task ON tool_playbook(task_type)")
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Cycle Checkpoints: retroactive UNIQUE constraint ──
    # checkpoints.py has UNIQUE in its CREATE TABLE, but databases created
    # before that change never get the constraint. ON CONFLICT upserts break.
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1 FROM pg_constraint
                WHERE conname = 'uq_checkpoint'
                OR conname = 'cycle_checkpoints_cycle_id_step_name_ticker_key'
            """)
            if not cur.fetchone():
                # Remove duplicates first (keep the newest)
                cur.execute("""
                    DELETE FROM cycle_checkpoints a USING (
                        SELECT MAX(ctid) as ctid, cycle_id, step_name, ticker
                        FROM cycle_checkpoints
                        GROUP BY cycle_id, step_name, ticker HAVING COUNT(*) > 1
                    ) b
                    WHERE a.cycle_id = b.cycle_id
                    AND a.step_name = b.step_name
                    AND a.ticker = b.ticker
                    AND a.ctid <> b.ctid
                """)
                cur.execute("""
                    ALTER TABLE cycle_checkpoints
                    ADD CONSTRAINT uq_checkpoint
                    UNIQUE (cycle_id, step_name, ticker)
                """)
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Ticker Validation ──
    _safe_add_column(conn, "discovered_tickers", "validation_status", "TEXT DEFAULT 'pending'")
    _safe_add_column(conn, "discovered_tickers", "rate_limited_count", "INTEGER DEFAULT 0")

    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ticker_quarantine (
                    ticker          TEXT PRIMARY KEY,
                    reason          TEXT NOT NULL,
                    details         TEXT,
                    quarantined_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass

    # ── Thesis tracking columns on analysis_results ──
    _safe_add_column(conn, "analysis_results", "thesis_verdict", "VARCHAR(10)")
    _safe_add_column(conn, "analysis_results", "thesis_confidence", "INTEGER")
    _safe_add_column(conn, "analysis_results", "thesis_summary", "TEXT")
    _safe_add_column(conn, "analysis_results", "thesis_updated_at", "TIMESTAMPTZ")
    _safe_add_column(
        conn, "analysis_results", "thesis_unchanged", "BOOLEAN DEFAULT FALSE"
    )

    # ── Heartbeat tracking on ticker_attention ──
    _safe_add_column(conn, "ticker_attention", "last_full_review_at", "TIMESTAMPTZ")

    # ── Watermarks table for delta collection ──
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ticker_collection_watermarks (
                    ticker          VARCHAR(10) NOT NULL,
                    source          VARCHAR(50) NOT NULL,
                    last_collected  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (ticker, source)
                )
            """)
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
