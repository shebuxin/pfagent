from openai import OpenAI
import os
import csv
import glob
import textwrap
import pandas as pd
import time
import subprocess

folder_path = "code_examples"
file_pattern = "*.py"
# Configure output directory and filename
output_directory = "generated_examples"  # Change this to your desired output directory
output_filename = "generated_output_summary.csv"
output_csv = os.path.join(output_directory, output_filename)

example_file = "examples.csv"

# api_key = os.getenv("OPENAI_API_KEY")

# if not api_key:
#     raise ValueError("Please Set Your OpenAI API Key")

client = OpenAI(api_key="")

file_paths = glob.glob(os.path.join(folder_path, file_pattern))
print(f"Found {len(file_paths)} Files to Process Matching Pattern '{file_pattern}' in '{folder_path}'")
print()

if not file_paths:
    print("No Files Found Matching the Specified Pattern")
    exit()

def execute_code(file_path):
    """Execute Python file using conda environment and return output"""
    try:
        # Execute the code using conda run
        result = subprocess.run(
            ["conda", "run", "-n", "power-simulation-env", "python", file_path],
            capture_output=True,
            text=True
        )
        
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        if result.returncode != 0:
            output += f"Return Code: {result.returncode}\n"
            
        return output if output else "No Output Generated"
        
    except subprocess.TimeoutExpired:
        return "Error: Code Execution Timed Out"
    except subprocess.CalledProcessError as e:
        return f"Error: Code Execution Failed with Return Code {e.returncode}\n{e.stderr}"
    except FileNotFoundError:
        return "Error: Conda Command Not Found"
    except Exception as e:
        return f"Error: Unexpected Error During Execution: {str(e)}"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)
print(f"Output Directory: {os.path.abspath(output_directory)}")
print(f"Output File: {os.path.abspath(output_csv)}")
print()

# Check if file exists and if it's empty
file_exists = os.path.exists(output_csv)
write_header = not file_exists or os.path.getsize(output_csv) == 0

# Write to CSV (append mode)
with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write header only if file is new or empty
    if write_header:
        csv_writer.writerow(['Question', 'Answer'])

    df = pd.read_csv(example_file)
    
    # Process each file
    total_entries = 0
    successful_files = 0
    failed_files = []
    
    for idx, code_path in enumerate(file_paths, 1):
        print(f"Processing File {idx}/{len(file_paths)}: {code_path}")
        
        try:
            system_prompt = textwrap.dedent(f"""
                You are a scientific assistant in electrical engineering. Your task is to summarize the output of code when provided with both the code and its results. Make sure your summary highlights information that is useful for an electrical engineer, grounded in both the code itself and the generated output.
            """).strip()

            with open(code_path, 'r', encoding='utf-8') as file:
                code = file.read()
            
            # Skip empty files
            if not code.strip():
                print(f"Skipping Empty File: {code_path}")
                continue
            
            # Execute the code and capture output
            print(f"Executing Code from {os.path.basename(code_path)}...")
            code_output = execute_code(code_path)
            print(f"Code Execution Completed for {os.path.basename(code_path)}")
            
            user_prompt = textwrap.dedent(f"""
                Code: {code}
                Code Output: {code_output}
            """).strip()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = client.chat.completions.create(
                model="o4-mini-2025-04-16",
                messages=messages
            )

            output = response.choices[0].message.content

            csv_writer.writerow([user_prompt, output])

            print(f"Generated Summary for {os.path.basename(code_path)}")
            total_entries += 1
            successful_files += 1
        
        except FileNotFoundError:
            print(f"Error: File '{code_path}' Not Found")
            failed_files.append(code_path)

        except Exception as e:
            print(f"Error Processing File {code_path}: {e}")
            failed_files.append(code_path)

print()
print(f"Total Files Processed Successfully: {successful_files}/{len(file_paths)}")
print(f"Total Entries Generated: {total_entries}")
print(f"Output Saved to: {os.path.abspath(output_csv)}")

if failed_files:
    print(f"Failed Files ({len(failed_files)}):")
    for failed_file in failed_files:
        print(f"\t - {failed_file}")