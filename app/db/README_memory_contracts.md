# Brain System: Memory & Observation Contracts

**Owner**: Developer 1 (Schema & Storage Owner)

This document serves as the single source of truth for the memory system's database schema and data-access layer (DAL). By adhering to these contracts, we prevent fragmentation and ensure consistent state management across the distributed multi-agent system.

## Non-Negotiable Integration Rules

1. **Shared Contracts Are Decided First**
   The tables `canonical_memories` and `episodic_observations` have been finalized. The field definitions within `schema.sql` MUST act as the absolute source of truth.

2. **File Ownership Matters**
   - Developer 1 owns: `app/db/schema.sql` and `app/services/memory/store.py`.
   - Modifying schema or database CRUD methods requires explicit sign-off from the Schema Owner.

3. **One Source of Truth per Concern**
   - **Schema**: `app/db/schema.sql` (End of file, under Brain System)
   - **DataAccess Helper**: `app/services/memory/store.py`
   - *DO NOT write inline SQL related to memory anywhere else in the application.*

4. **No Direct Writes to Memory Except Through Helper**
   - To add an observation, you MUST use `MemoryStore.add_episodic_observation()`.
   - To promote memory, you MUST use `MemoryStore.add_canonical_memory()` alongside `mark_observation_promoted()`.
   - Do not bypass this API with a direct JDBC/ODBC or `psycopg` execution.

## Data Schema Summary

### Canonical Memory (`canonical_memories`)
Store of curated, long-term lessons.

- **ID**: UUID string
- **type**: `VARCHAR` (e.g. `market_pattern`, `ticker_quirk`, etc.)
- **tags**: JSON String representing the list of tags (Deserialized lazily by DAL).
- **confidence_score**: `DOUBLE` Float. 
- **status**: Defaults to `tentative`. Changes to `active` or `deprecated` on review.

### Episodic Observation (`episodic_observations`)
Raw, short-term occurrences waiting to be consolidated.

- **cycle_id**: Link back to the run pipeline ID.
- **source_type**: `VARCHAR` (`decision`, `outcome`, `postmortem`, `manual`)
- **promoted_to_memory**: `BOOLEAN` indicating consolidation.

All fields are fully mapped in `app/services/memory/store.py` within the `MemoryStore` utility class.
