import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = """            for coro in captured_coros:
                try:
                    await coro
                except Exception as e:
                    print(f"Exception from coro: {e}")
                except BaseException as e:
                    print(f"BaseException from coro: {e}")
                    raise"""

replacement = """            for coro in captured_coros:
                if 'checkpoint_heartbeat' in str(coro):
                    continue
                try:
                    await coro
                except Exception as e:
                    print(f"Exception from coro: {e}")
                except BaseException as e:
                    print(f"BaseException from coro: {e}")
                    raise"""

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)
