with open("tests/test_general_audit.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.endswith("\\\\\n"):
        new_lines.append(line[:-3] + "\\\n")
    elif line.endswith("\\\\ \n"):
        new_lines.append(line[:-4] + "\\\n")
    else:
        new_lines.append(line)

with open("tests/test_general_audit.py", "w") as f:
    f.writelines(new_lines)
