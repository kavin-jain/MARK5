import os
import ast
import traceback
import sys

def audit_directory(root_dir):
    print(f"🔍 Starting Syntax Audit of {root_dir}...")
    
    error_count = 0
    file_count = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip pycache, venv, git
        if "__pycache__" in dirpath or ".venv" in dirpath or ".git" in dirpath:
            continue
            
        for filename in filenames:
            if filename.endswith(".py"):
                file_count += 1
                full_path = os.path.join(dirpath, filename)
                
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    
                    # 1. AST Parse (Syntax Check)
                    ast.parse(source, filename=full_path)
                    
                    # 2. Compilation Check (Runtime Syntax)
                    compile(source, full_path, 'exec')
                    
                except SyntaxError as e:
                    print(f"❌ SYNTAX ERROR in {full_path}:")
                    print(f"   Line {e.lineno}: {e.text.strip() if e.text else ''}")
                    print(f"   {e.msg}")
                    error_count += 1
                except Exception as e:
                    print(f"⚠️  READ ERROR in {full_path}: {e}")
                    error_count += 1

    print("-" * 40)
    if error_count == 0:
        print(f"✅ Success! Scanned {file_count} files. No Syntax Errors found.")
        sys.exit(0)
    else:
        print(f"❌ Failed! Found {error_count} errors in {file_count} files.")
        sys.exit(1)

if __name__ == "__main__":
    audit_directory("core")
