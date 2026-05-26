#!/usr/bin/env python3
import os
import sys
import subprocess
import time

def main():
    test_dir = "tests"
    if not os.path.isdir(test_dir):
        print(f"Error: {test_dir} directory not found.")
        sys.exit(1)

    # Walk and collect all test files
    test_files = []
    for root, _, files in os.walk(test_dir):
        # Exclude pycache
        if "__pycache__" in root:
            continue
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                test_files.append(os.path.join(root, file))

    test_files.sort()
    total = len(test_files)
    print(f"Found {total} test files to execute sequentially.")
    print("Each file has a 90-second timeout to prevent hangs.")
    print("=" * 60)

    passed_files = []
    failed_files = []
    timed_out_files = []

    # Get env
    env = os.environ.copy()
    env["TRADING_BOT_TEST_DB"] = "1"

    start_time = time.time()

    for idx, file_path in enumerate(test_files, 1):
        print(f"[{idx}/{total}] Running: {file_path} ...", end="", flush=True)
        file_start = time.time()
        try:
            # Run pytest in a subprocess
            res = subprocess.run(
                [".venv/bin/pytest", file_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=90,
                text=True
            )
            elapsed = time.time() - file_start
            if res.returncode == 0:
                print(f" \033[92mPASSED\033[0m ({elapsed:.1f}s)")
                passed_files.append((file_path, elapsed))
            else:
                print(f" \033[91mFAILED\033[0m ({elapsed:.1f}s)")
                failed_files.append((file_path, elapsed, res.stdout, res.stderr))
        except subprocess.TimeoutExpired:
            elapsed = time.time() - file_start
            print(f" \033[93mTIMEOUT (HUNG)\033[0m ({elapsed:.1f}s)")
            timed_out_files.append((file_path, elapsed))

    duration = time.time() - start_time
    print("=" * 60)
    print(f"Execution finished in {duration:.1f}s.")
    print(f"PASSED: {len(passed_files)}")
    print(f"FAILED: {len(failed_files)}")
    print(f"TIMED OUT: {len(timed_out_files)}")

    if failed_files:
        print("\nFailed Test Files Details:")
        for file_path, elapsed, stdout, stderr in failed_files:
            print(f"- {file_path} ({elapsed:.1f}s)")
            # Print short snippet of output
            lines = stdout.splitlines() + stderr.splitlines()
            snippet = [line for line in lines if "FAILED" in line or "AssertionError" in line or "Exception" in line or "Error" in line]
            if snippet:
                print("  Key Errors:")
                for s in snippet[-5:]:
                    print(f"    {s}")
            else:
                print("  Last 5 lines of stdout:")
                for s in lines[-5:]:
                    print(f"    {s}")

    if timed_out_files:
        print("\nHung/Timed Out Test Files:")
        for file_path, elapsed in timed_out_files:
            print(f"- {file_path} ({elapsed:.1f}s)")

    if failed_files or timed_out_files:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
