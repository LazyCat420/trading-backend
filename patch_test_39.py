with open("tests/test_general_audit.py", "r") as f:
    lines = f.readlines()

in_target = False
new_lines = []
skip = False
for i, line in enumerate(lines):
    if "ctx.cycle_id = \"cycle_timeout\"" in line:
        new_lines.append(line)
        new_lines.append("        ctx.tickers = [\"AAPL\"]\n")
        new_lines.append("        ctx.analyze = True\n")
        continue
    
    if "results = await run_phase4_analysis(" in line:
        in_target = True
        new_lines.append(line)
        continue
        
    if in_target and "tickers=[\"AAPL\"]," in line:
        continue # remove it
        
    if in_target and "emit=MagicMock()," in line:
        new_lines.append("                macro_memo=\"\",\n")
        new_lines.append(line)
        continue
        
    if in_target and ")" in line and "state={}" not in line:
        in_target = False
        
    new_lines.append(line)

with open("tests/test_general_audit.py", "w") as f:
    f.writelines(new_lines)
