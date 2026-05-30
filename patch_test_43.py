with open("tests/test_general_audit.py", "r") as f:
    content = f.read()

content = content.replace("}), \\\\\n", "}), \\\n")
content = content.replace("}), \\\\\n", "}), \\\n")
content = content.replace("), \\\\\n", "), \\\n")
content = content.replace("], \\\\\n", "], \\\n")

with open("tests/test_general_audit.py", "w") as f:
    f.write(content)
