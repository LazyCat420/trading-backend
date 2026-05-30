import os
import ast

def find_indexing(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content, filename=filepath)
    except SyntaxError:
        return []

    findings = []
    
    # We walk the AST to find Subscript nodes where the slice is Constant(value=0) or Constant(value=-1)
    # (or Index with Num/UnaryOp in Python <3.9)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            # Check the index
            is_zero_or_neg_one = False
            sl = node.slice
            if isinstance(sl, ast.Constant):
                if sl.value in (0, -1):
                    is_zero_or_neg_one = True
            elif isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.USub):
                if isinstance(sl.operand, ast.Constant) and sl.operand.value == 1:
                    is_zero_or_neg_one = True
            
            if is_zero_or_neg_one:
                # Get the node line number
                lineno = getattr(node, 'lineno', 0)
                # print node value / slice to see context
                try:
                    expr_src = ast.unparse(node)
                except Exception:
                    expr_src = "unknown"
                findings.append((filepath, lineno, expr_src))
                
    return findings

def main():
    target_dirs = ['app/cycle', 'app/pipeline', 'app/trading']
    all_findings = []
    for t_dir in target_dirs:
        if not os.path.exists(t_dir):
            continue
        for root, dirs, files in os.walk(t_dir):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    all_findings.extend(find_indexing(filepath))
                    
    print(f"Found {len(all_findings)} index lookups:")
    for file, line, expr in all_findings:
        print(f"{file}:{line} -> {expr}")

if __name__ == "__main__":
    main()
