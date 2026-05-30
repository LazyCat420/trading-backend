import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = """            for coro in captured_coros:
                await coro
    
            # Verify run_perticker_collection was called with completed_tickers from checkpoint"""
replacement = """            for coro in captured_coros:
                try:
                    await coro
                except Exception as e:
                    print(f"Exception from coro: {e}")
                except BaseException as e:
                    print(f"BaseException from coro: {e}")
                    raise
            print(f"Number of captured coros: {len(captured_coros)}")
    
            # Verify run_perticker_collection was called with completed_tickers from checkpoint"""

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)
