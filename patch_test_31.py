with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

import re
content = re.sub(r'garbage = "```json\n', 'garbage = "```json\\\\n{ oops }\\\\n```"', content)

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)
