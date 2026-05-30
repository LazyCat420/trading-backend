import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = 'tools = [{"name": "get_stock_price"}, {"function": {"name": "buy_stock"}}]'
replacement = 'tools = [{"name": "get_stock_price"}, {"type": "function", "function": {"name": "buy_stock"}}]'

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)
