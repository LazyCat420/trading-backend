# Issue: Verify and Enable META_AGENT_ENABLED (Phase 6 Rollout) in Production

## Description
The `META_AGENT_ENABLED` feature flag is currently gated and defaults to `False` in `Settings`. This controls whether the autonomous Meta-Agent runs periodically (every 6 hours) to look at trading logs, performance reviews, and automatically adapt / generate new instruction lenses (prompts) to refine the trading strategy.

Before enabling this flag in production, we need to verify Phase 6 execution safety, ensure prompt generation doesn't drift, and cap prompt generation limits to avoid context blowouts.

## Tasks
- [ ] Confirm Phase 6 learning loop runs successfully in a staging environment.
- [ ] Verify that generated prompts are properly evaluated and benched if they underperform.
- [ ] Monitor memory and token usage of the Meta-Agent runs to prevent cost spikes on vLLM.
- [ ] Once verified, set `META_AGENT_ENABLED = True` in `app/config/config.py`.
