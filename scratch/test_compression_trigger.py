import os
import sys
import asyncio

# Add parent directory to path to allow importing app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.context_compressor import compress_history
from app.config.context_budget import get_context_budget

async def test_compression():
    print("=" * 80)
    print("TESTING HISTORY COMPRESSION TRIGGER")
    print("=" * 80)

    # 1. Setup a dummy message history that exceeds the default threshold (24,576 tokens / ~98k chars)
    # Default budget compressor threshold:
    budget = get_context_budget()
    threshold = budget.compressor_threshold
    print(f"Default Model: {budget.model_id}")
    print(f"Effective Context: {budget.effective_context_tokens} tokens")
    print(f"Compressor Threshold: {threshold} tokens (~{threshold * 4} characters)")

    # Construct messages: system prompt + some middle turns + keep_recent turns
    messages = [
        {"role": "system", "content": "You are a helpful analyst. Keep track of all calculations."},
    ]

    # Add 10 turns of heavy data to exceed the threshold
    # Each turn will have ~3,000 tokens (approx 12,000 characters)
    for i in range(10):
        messages.extend([
            {"role": "user", "content": f"Turn {i}: Here is some heavy market data for analysis: " + "A" * 6000},
            {"role": "assistant", "content": f"Understood. I have recorded the details for turn {i}.", "tool_calls": []},
            {"role": "tool", "name": "precise_calculator", "content": f"Calculated metrics for turn {i}: " + "B" * 6000}
        ])

    # Add the recent turns that should be kept
    messages.extend([
        {"role": "user", "content": "What is the final consensus?"},
        {"role": "assistant", "content": "Based on the above 10 turns of data, the consensus is positive.", "tool_calls": []},
        {"role": "user", "content": "Verify that last calculation."},
        {"role": "assistant", "content": "Verifying calculation now...", "tool_calls": []},
        {"role": "user", "content": "What is the final score?"}
    ])

    total_tokens_before = sum(len(m.get("content", "")) // 4 for m in messages)
    print(f"\nCreated history with {len(messages)} messages.")
    print(f"Total estimated tokens before compression: {total_tokens_before}")

    # 2. Run compress_history
    print("\nRunning compress_history...")
    try:
        compressed_messages = await compress_history(messages, keep_recent=3)
        
        total_tokens_after = sum(len(m.get("content", "")) // 4 for m in compressed_messages)
        print(f"\nCompression complete! New history has {len(compressed_messages)} messages.")
        print(f"Total estimated tokens after compression: {total_tokens_after}")
        
        # Verify the structure of the compressed messages
        print("\n--- Message Structure after Compression ---")
        for i, m in enumerate(compressed_messages):
            content_preview = m.get("content", "")[:100]
            if len(m.get("content", "")) > 100:
                content_preview += "..."
            print(f"Message {i} [{m.get('role')}]: {content_preview}")
            
        # Verify that we have the compressed history summary message
        has_summary = any("[COMPRESSED HISTORY SUMMARY]" in m.get("content", "") for m in compressed_messages)
        if has_summary:
            print("\n✅ SUCCESS: Context compressor triggered and generated history summary successfully!")
        else:
            print("\n❌ FAILURE: Context compressor did not generate the summary message.")
            
    except Exception as e:
        print(f"\n❌ ERROR running compress_history: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_compression())
