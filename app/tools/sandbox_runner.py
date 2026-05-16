import sys


def run_isolated(code: str):
    try:
        import RestrictedPython
        from RestrictedPython import compile_restricted
        from RestrictedPython.Guards import safe_builtins, full_write_guard
        from RestrictedPython.Eval import (
            default_guarded_getitem,
            default_guarded_getiter,
        )
    except ImportError as e:
        print(f"Error: RestrictedPython is not installed or failed to import: {e}")
        return

    my_globals = {
        "__builtins__": safe_builtins,
        "_print_": RestrictedPython.PrintCollector,
        "_getattr_": getattr,
        "_getitem_": default_guarded_getitem,
        "_getiter_": default_guarded_getiter,
        "_write_": full_write_guard,
        "math": __import__("math"),
        "json": __import__("json"),
    }

    try:
        byte_code = compile_restricted(code, "<string>", "exec")
    except Exception as e:
        print(f"Compilation Error (Safety Violation): {e}")
        return

    loc = {}
    try:
        exec(byte_code, my_globals, loc)
        if "_print" in loc and loc["_print"]:
            print(loc["_print"]())
        else:
            print("Code executed successfully with no print output.")
    except Exception as e:
        print(f"Runtime Error: {e}")


if __name__ == "__main__":
    code = sys.stdin.read()
    run_isolated(code)
