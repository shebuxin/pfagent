import os
import json
import glob
import pandas as pd

code_folder_path = "code_examples"
code_file_pattern = "*.py"
csv_folder_path = "generated_examples"
csv_file_pattern = "*csv"

output_jsonl = "fine_tuning_data.jsonl"

code_file_paths = glob.glob(os.path.join(code_folder_path, code_file_pattern))
print(f"Found {len(code_file_paths)} Code Files Matching Pattern '{code_file_pattern}' in '{code_folder_path}'")
print()

csv_file_paths = glob.glob(os.path.join(csv_folder_path, csv_file_pattern))
print(f"Found {len(csv_file_paths)} CSV Files Matching Pattern '{csv_file_pattern}' in '{csv_folder_path}'")
print()

if not code_file_paths or not csv_file_paths:
    print("No Files Found Matching the Specified Pattern")
    exit()

# Create a mapping of code filenames to full paths for easier lookup
code_file_map = {}
for code_path in code_file_paths:
    filename = os.path.basename(code_path)
    code_file_map[filename] = code_path

# Process each CSV file
total_entries = 0
successful_files = 0
failed_files = []

with open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
    
    for idx, csv_path in enumerate(csv_file_paths, 1):
        print(f"Processing CSV File {idx}/{len(csv_file_paths)}: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            if 'Question' not in df.columns:
                print(f"Warning: CSV File {csv_path} Missing Required 'Question' Column")
                continue
                
            # Determine CSV format
            has_file_path = 'File Path to Answer' in df.columns
            has_direct_answer = 'Answer' in df.columns
            
            if not has_file_path and not has_direct_answer:
                print(f"Warning: CSV File {csv_path} Missing Both 'File Path to Answer' and 'Answer' Columns")
                continue
            
            csv_type = "File Path" if has_file_path else "Direct Answer"
            print(f"Processing CSV with {csv_type} Format")
            
            # Process each row in the CSV
            for row_idx, row in df.iterrows():
                question = row['Question']
                
                if pd.isna(question):
                    continue
                
                try:
                    assistant_response = None
                    
                    # Handle CSV with direct answers
                    if has_direct_answer and not pd.isna(row['Answer']):
                        assistant_response = str(row['Answer'])
                        print(f"Using Direct Answer for Row {row_idx + 1}")
                    
                    # Handle CSV with file paths (only if no direct answer or direct answer is empty)
                    elif has_file_path and not pd.isna(row['File Path to Answer']):
                        code_file_ref = row['File Path to Answer']
                        
                        # Find the corresponding code file
                        code_file_path = None
                        
                        # First try exact filename match
                        if code_file_ref in code_file_map:
                            code_file_path = code_file_map[code_file_ref]
                        else:
                            # Try to find by partial match or full path
                            for code_path in code_file_paths:
                                if code_file_ref in code_path or os.path.basename(code_path) == code_file_ref:
                                    code_file_path = code_path
                                    break
                        
                        if not code_file_path:
                            print(f"Warning: Code File '{code_file_ref}' Not Found for Question in Row {row_idx + 1}")
                            continue
                        
                        with open(code_file_path, 'r', encoding='utf-8') as file:
                            code = file.read()
                        
                        if not code.strip():
                            print(f"Skipping Empty Code File: {code_file_path}")
                            continue
                        
                        assistant_response = code
                        print(f"Using Code Content for Row {row_idx + 1}: {os.path.basename(code_file_path)}")
                    
                    else:
                        print(f"Warning: No Valid Answer Source Found for Question in Row {row_idx + 1}")
                        continue
                    
                    # Create JSONL entry
                    jsonl_entry = {
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": assistant_response}
                        ]
                    }
                    
                    # Write to JSONL file
                    jsonl_file.write(json.dumps(jsonl_entry) + '\n')
                    total_entries += 1
                
                except FileNotFoundError:
                    print(f"Error: Code File Not Found for Row {row_idx + 1}")
                    failed_files.append(f"{csv_path}:row_{row_idx + 1}")

                except Exception as e:
                    print(f"Error Processing Row {row_idx + 1}: {e}")
                    failed_files.append(f"{csv_path}:row_{row_idx + 1}")

            successful_files += 1
        
        except Exception as e:
            print(f"Error Processing CSV File {csv_path}: {e}")
            failed_files.append(csv_path)

print()
print(f"Total CSV Files Processed Successfully: {successful_files}/{len(csv_file_paths)}")
print(f"Total JSONL Entries Generated: {total_entries}")
print(f"Output Saved to: {output_jsonl}")

if failed_files:
    print(f"Failed Items ({len(failed_files)}):")
    for failed_item in failed_files:
        print(f"\t - {failed_item}")