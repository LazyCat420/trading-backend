import sys

try:
    from app.db.connection import get_db

    db = get_db()
    db.execute("DELETE FROM sec_13f_holdings")
    print("[db] Deleted all rows from sec_13f_holdings")
    db.execute(
        "UPDATE sec_13f_filers SET latest_quarter = NULL, next_expected_filing = NULL"
    )
    print("[db] Reset sec_13f_filers metadata")
    print("Done")
except Exception as e:
    import traceback

    traceback.print_exc()
    sys.exit(2)
