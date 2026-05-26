#!/usr/bin/env python3
"""
Evaluation Script: Smart Janitor & Summarizer Flow.
Simulates the data pipeline, runs the Smart Janitor on test fixtures,
formats the summary from the qualitative draft, and uses an LLM-based judge
to evaluate the summary against the original text (Ground Truth) for:
1. Factual Grounding (no hallucinations)
2. Factual Retention (accuracy/preservation of key metrics)
"""

import sys
import os
import asyncio
import json
import logging
from datetime import datetime

# Adjust Python path to load app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.prism_agent_caller import call_prism_agent
from app.processors.smart_janitor import SMART_JANITOR_SYSTEM_PROMPT
from app.services.vllm_client import Priority

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_janitor_summarizer")

# Test fixtures representing raw scraped articles
FIXTURES = [
    {
        "id": "art_001_hims",
        "ticker": "HIMS",
        "title": "Hims & Hers Health Inc. Reports Q1 2026 Earnings: Revenue Skyrockets 46% YoY to $382.4 Million",
        "publisher": "Yahoo Finance",
        "published_at": datetime.now(),
        "summary": """Hims & Hers Health, Inc. (NYSE: HIMS) today reported financial results for the first quarter ended March 31, 2026. 
The company posted outstanding growth, with Q1 revenue increasing 46% year-over-year to $382.4 million, compared to $261.9 million in Q1 2025. 
Net income reached $28.3 million, up from $8.5 million in the prior year quarter. Diluted EPS was $0.13, beating consensus estimates of $0.09.
CEO Andrew Dudum commented: 'Our expansion into personalized medicine, particularly weight management and cardiovascular health, has exceeded our expectations. We added 142,000 net new subscribers in Q1, bringing our total subscriber base to 1.82 million.'
For the full year 2026, Hims & Hers raised its revenue guidance to between $1.52 billion and $1.55 billion, representing 32% to 35% growth YoY, up from its previous guidance of $1.48 billion to $1.51 billion."""
    },
    {
        "id": "art_002_adbe",
        "ticker": "HIMS",  # Wrong ticker context (should be ADBE) to test routing/discard behavior
        "title": "Adobe Launches Firefly V3 Image Generator at Summit with Real-Time Video Capabilities",
        "publisher": "TechCrunch",
        "published_at": datetime.now(),
        "summary": """Adobe (NASDAQ: ADBE) today unveiled Firefly V3, the latest generation of its generative AI model family. 
Speaking at the annual Adobe Summit, CEO Shantanu Narayen announced that Firefly V3 is now capable of producing high-definition images in under 1.2 seconds, 
and introduces real-time video generation features for Creative Cloud subscribers. 
'We are putting enterprise-grade, copyright-safe generative AI directly into the hands of millions of creators,' Narayen said. 
Adobe shares closed up 2.4% following the announcement, although some analysts remain cautious about competition from OpenAI's Sora and Midjourney V6."""
    },
    {
        "id": "art_003_spam",
        "ticker": "HIMS",
        "title": "10 Hidden Stocks That Could Make You Rich in 2026! (Number 7 Will Shock You!)",
        "publisher": "PennyStockHype",
        "published_at": datetime.now(),
        "summary": """Are you tired of small returns on boring blue chip stocks? The stock market is entering a massive bull run and you don't want to be left behind.
We have scanned the entire market and found 10 micro-cap stocks poised to shoot to the moon. Stocks like Hims and others are being talked about in Reddit chatrooms. 
Don't wait for Wall Street to buy these up. Subscribe to our premium newsletter today for just $49/month to get the full list and secure your financial freedom! This is not financial advice."""
    }
]

JUDGE_PROMPT = """You are an independent QA Auditor for a quant hedge fund. 
Your task is to compare a generated summary against the raw text (Ground Truth) of an article and grade the summary on two criteria:

1. FACTUAL GROUNDING (0-100): 
- Are there any claims, numbers, dates, or names in the summary that are NOT in the original text (i.e. hallucinations)? 
- Deduct 20 points for each hallucinated claim. If there are no hallucinations, score is 100.

2. INFORMATION RETENTION (0-100):
- Does the summary capture the core news event and key quantitative metrics (e.g. revenue figures, percentage changes, names, guidance numbers) accurately?
- Deduct points for missing critical metrics or vague statements that strip out useful quantitative data.

Format your response strictly as a JSON object:
{{
  "factual_grounding_score": 0-100,
  "factual_grounding_reason": "Explanation of any hallucinations or '100 - Perfect grounding'",
  "information_retention_score": 0-100,
  "information_retention_reason": "Explanation of missing metrics or '100 - All key metrics retained'",
  "overall_critique": "Short overall critique of the summary quality."
}}

Original Text (Ground Truth):
\"\"\"
{original}
\"\"\"

Generated Summary:
\"\"\"
{summary}
\"\"\"
"""


async def run_smart_janitor_mock(fixture: dict) -> dict:
    """Simulate the Smart Janitor Agent run."""
    user_message = f"""Ticker: {fixture['ticker']}
Source: News Article
Publisher: {fixture['publisher']}
Published At: {fixture['published_at'].isoformat()}
Title: {fixture['title']}
Summary/Snippet: {fixture['summary']}
"""
    logger.info(f"Running Smart Janitor on: {fixture['title'][:40]}...")
    
    response, _, _ = await call_prism_agent(
        agent_id="CUSTOM_SYSTEM_JANITOR_AGENT",
        user_message=user_message,
        fallback_system_prompt=SMART_JANITOR_SYSTEM_PROMPT,
        fallback_agent_name="smart_janitor",
        temperature=0.1,
        max_tokens=1024,
        priority=Priority.NORMAL,
        ticker=fixture['ticker']
    )
    
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
        
    return json.loads(cleaned)


def format_summary_from_draft(draft: dict, title: str) -> str:
    """Format the final llm_summary from the qualitative draft (Optimized summarizer logic)."""
    decision = draft.get("decision", "keep").lower()
    if decision == "discard":
        return ""
        
    bullets = draft.get("bullet_points", [])
    if isinstance(bullets, list):
        summary_text = "\n".join(
            f"- {str(b)}" if not str(b).strip().startswith("-") else str(b)
            for b in bullets
        )
    else:
        summary_text = str(bullets)
        
    if not summary_text.strip():
        summary_text = draft.get("justification", "") or title
        
    return summary_text


async def evaluate_summary_quality(original: str, summary: str) -> dict:
    """Use the LLM Judge to grade the summary against the original text."""
    logger.info("Evaluating summary quality with LLM Judge...")
    
    response, _, _ = await call_prism_agent(
        agent_id="CUSTOM_DATA_JANITOR_CRITIC_AGENT",
        user_message=JUDGE_PROMPT.format(original=original, summary=summary),
        fallback_system_prompt="You are a strict QA API. Respond ONLY in JSON.",
        fallback_agent_name="qa_critic",
        temperature=0.1,
        max_tokens=500,
        priority=Priority.NORMAL
    )
    
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
        
    return json.loads(cleaned)


async def main():
    print("=" * 80)
    print("STARTING DATA PIPELINE PIPELINE OPTIMIZATION EVALUATION")
    print("=" * 80)
    
    results = []
    
    for fixture in FIXTURES:
        print(f"\n>>> Processing Fixture: {fixture['id']} ({fixture['ticker']})")
        print(f"Title: {fixture['title']}")
        
        start_time = datetime.now()
        
        # 1. Run Smart Janitor
        try:
            draft = await run_smart_janitor_mock(fixture)
        except Exception as e:
            print(f"Error running Smart Janitor: {e}")
            continue
            
        decision = draft.get("decision", "keep")
        print(f"Decision: {decision.upper()}")
        print(f"Theme: {draft.get('suggested_theme', 'N/A')}")
        print(f"Justification: {draft.get('justification', 'N/A')}")
        
        # 2. Format Summary
        summary = format_summary_from_draft(draft, fixture['title'])
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if decision == "discard":
            print("Article discarded. Skipping summary generation and LLM Judge evaluation.")
            results.append({
                "id": fixture["id"],
                "decision": "discard",
                "elapsed": elapsed,
                "draft": draft,
                "summary": "",
                "evaluation": None
            })
            continue
            
        print("\nGenerated Summary (from qualitative draft):")
        print("-" * 50)
        print(summary)
        print("-" * 50)
        
        # 3. LLM Judge evaluation against original (Ground Truth)
        try:
            evaluation = await evaluate_summary_quality(fixture['summary'], summary)
        except Exception as e:
            print(f"Error running LLM Judge: {e}")
            evaluation = {}
            
        print("\nEvaluation Results:")
        print(f"  Factual Grounding (No Hallucinations): {evaluation.get('factual_grounding_score', 'N/A')}/100")
        print(f"  Reason: {evaluation.get('factual_grounding_reason', 'N/A')}")
        print(f"  Information Retention (Accuracy):     {evaluation.get('information_retention_score', 'N/A')}/100")
        print(f"  Reason: {evaluation.get('information_retention_reason', 'N/A')}")
        print(f"  Overall Critique: {evaluation.get('overall_critique', 'N/A')}")
        
        results.append({
            "id": fixture["id"],
            "decision": "keep",
            "elapsed": elapsed,
            "draft": draft,
            "summary": summary,
            "evaluation": evaluation
        })
        
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    # Estimate token/LLM call savings
    llm_calls_saved = len(FIXTURES)  # Bypassed 1 summarizer call per fixture
    # Add disabled simple janitor background task savings (if it would have run)
    llm_calls_saved += len(FIXTURES)
    
    print(f"Total Fixtures Evaluated: {len(FIXTURES)}")
    print(f"Redundant LLM Calls Bypassed: {llm_calls_saved}")
    print(f"Total Computation Saved: ~{llm_calls_saved * 1500} Input Tokens + ~{llm_calls_saved * 400} Output Tokens per cycle")
    print("\nResults by Fixture:")
    for res in results:
        eval_part = ""
        if res["decision"] == "keep" and res["evaluation"]:
            eval_part = f" | Grounding: {res['evaluation'].get('factual_grounding_score')}/100 | Retention: {res['evaluation'].get('information_retention_score')}/100"
        print(f"  - {res['id']}: Decision={res['decision'].upper()} | Time={res['elapsed']:.2f}s{eval_part}")
        
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
