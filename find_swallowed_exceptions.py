import os
import ast

def is_logging_or_raising(node):
    # Check if the handler contains a raise statement, a logging call, or trace print
    has_raise = False
    has_logging = False
    
    for subnode in ast.walk(node):
        if isinstance(subnode, ast.Raise):
            has_raise = True
        elif isinstance(subnode, ast.Call):
            # Check for logger.xxxx, logging.xxxx, self.logger.xxxx
            func = subnode.func
            func_name = ""
            if isinstance(func, ast.Attribute):
                func_name = func.attr
                # check if the root is logger or self.logger
                curr = func.value
                while isinstance(curr, ast.Attribute):
                    curr = curr.value
                if isinstance(curr, ast.Name) and curr.id in ('logger', 'logging', 'self'):
                    has_logging = True
            elif isinstance(func, ast.Name):
                if func.id == 'print':
                    # sometimes prints are used, though we want actual loggers
                    has_logging = True
    return has_raise or has_logging

def check_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content, filename=filepath)
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return []

    findings = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                # check if handler catches Exception, BaseException, or is bare except:
                is_target = False
                if handler.type is None:
                    # bare except
                    is_target = True
                    exc_name = "bare except"
                elif isinstance(handler.type, ast.Name) and handler.type.id in ('Exception', 'BaseException'):
                    is_target = True
                    exc_name = handler.type.id
                elif isinstance(handler.type, ast.Attribute) and handler.type.attr in ('Exception', 'BaseException'):
                    is_target = True
                    exc_name = handler.type.attr
                
                if is_target:
                    # Let's inspect the body
                    body = handler.body
                    # check if the body is just pass or doesn't log/raise
                    if not is_logging_or_raising(handler):
                        # Get source line range
                        start_line = handler.lineno
                        # Check what statements are inside
                        statements = [type(stmt).__name__ for stmt in body]
                        findings.append((filepath, start_line, exc_name, statements))
                        
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
                    all_findings.extend(check_file(filepath))
                    
    print(f"Found {len(all_findings)} potentially swallowed exception blocks:")
    for file, line, exc_type, stmts in all_findings:
        print(f"{file}:{line} catches '{exc_type}', body stmts: {stmts}")

if __name__ == "__main__":
    main()
