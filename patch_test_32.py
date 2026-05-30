import re

with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

target = '        garbage = "```json\\n{ oops }\\n```"{ oops }\n```"'
replacement = '        garbage = "```json\\n{ oops }\\n```"'

content = content.replace(target, replacement)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)
