#!/usr/bin/env python3

import subprocess
import sys
import os

def run_script(script_name):
    """Run a Python script and handle errors."""
    print(f"Running {script_name}...")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              check=True, 
                              capture_output=True, 
                              text=True)
        print(f"{script_name} Completed Successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except Exception as e:
        print(f"Unexpected Error Running {script_name}: {e}")
        return False

def main():
    """Main function to run all scripts in sequence."""
    
    scripts = [
        "high_level_generate_qa_pairs.py",
        "low_level_generate_qa_pairs.py", 
        "summary_generation.py",
        "fine_tuning_json_generation.py",
        "finetune.py"
    ]
    
    print("Starting Sequential Script Execution...\n")
    
    for i, script in enumerate(scripts, 1):
        print(f"Step {i}: Running {script}")
        
        if not os.path.exists(script):
            print(f"Error: {script} Not Found in Current Directory")
            sys.exit(1)
            
        success = run_script(script)
        
        if not success:
            print(f"\nPipeline Failed at {script}")
            sys.exit(1)
        
        print()
    
    print("All Scripts Completed Successfully")

if __name__ == "__main__":
    main()