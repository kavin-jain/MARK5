import subprocess
import sys
import os

def run_test(iteration):
    print(f"Running iteration {iteration}/10...")
    # Use absolute path for the test script
    test_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_health_check.py")
    
    # Run the test script using the current python interpreter
    result = subprocess.run([sys.executable, test_script], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Iteration {iteration} PASSED.")
        return True
    else:
        print(f"Iteration {iteration} FAILED.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False

def main():
    success_count = 0
    for i in range(1, 11):
        if run_test(i):
            success_count += 1
        else:
            print("Stress test aborted due to failure.")
            break
    
    print(f"\nFinal Result: {success_count}/10 iterations passed.")
    if success_count == 10:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
