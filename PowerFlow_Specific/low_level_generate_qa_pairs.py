# Layered domain knowledge distilliation -- low level

from openai import OpenAI
from pydantic import BaseModel
from typing import List
import os
import csv
import glob
import textwrap
import pandas as pd
import time

class OutputFormat(BaseModel):
    prompts: List[str] 

folder_path = "code_examples"
file_pattern = "*.py"

# Configure output directory and filename
output_directory = "generated_examples"  # Change this to your desired output directory
output_filename = "low_level_generated_prompts.csv"
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
        csv_writer.writerow(['Question', 'File Path to Answer'])

    df = pd.read_csv(example_file)
    
    # Process each file
    total_prompts = 0
    successful_files = 0
    failed_files = []
    
    for idx, code_path in enumerate(file_paths, 1):
        print(f"Processing File {idx}/{len(file_paths)}: {code_path}")
        
        try:
            example_prompt = df[df['Code File'] == os.path.basename(code_path)]['Task']
            
            if not example_prompt.empty:
                system_prompt = textwrap.dedent(f"""
                    You are an expert in AI training data generation. Your task is to create natural, clear, varied, and tonally diverse questions or instructions that a user might ask to produce (roughly) the given code as the output. Vary the length and level of detail — some prompts should be highly specific and detailed, while others can be brief or vague, with a greater emphasis on generating more vague prompts. Since the goal is to create natural prompts that a typical user would ask, avoid overly technical or exhaustive specifications (e.g., listing exact parameter values to set to true/false) unless it would be realistic for a normal user. Do not include any labels, annotations, or descriptions before the prompts (e.g., “Prompt: …” or “Short prompt: …”). Respond only with the list of generated prompts. These questions will be used to create high-quality Q&A pairs for fine-tuning a code-generation model.

                    Example Question Prompt: {example_prompt}
                """).strip()
            else:
                system_prompt = "You are an expert in AI training data generation. Your task is to create natural, clear, varied, and tonally diverse questions or instructions that a user might ask to produce (roughly) the given code as the output. Vary the length and level of detail — some prompts should be highly specific and detailed, while others can be brief or vague, with a greater emphasis on generating more vague prompts. Since the goal is to create natural prompts that a typical user would ask, avoid overly technical or exhaustive specifications (e.g., listing exact parameter values to set to true/false) unless it would be realistic for a normal user. Do not include any labels, annotations, or descriptions before the prompts (e.g., “Prompt: …” or “Short prompt: …”). Respond only with the list of generated prompts. These questions will be used to create high-quality Q&A pairs for fine-tuning a code-generation model."
                print("No Example Prompt Found")

            with open(code_path, 'r', encoding='utf-8') as file:
                code = file.read()
            
            # Skip empty files
            if not code.strip():
                print(f"Skipping Empty File: {code_path}")
                continue
            
            user_prompt = f"{code}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = client.chat.completions.parse(
                model="o4-mini-2025-04-16",
                messages=messages,
                response_format=OutputFormat,
            )

            # print(response.choices[0].message.parsed)

            prompts = response.choices[0].message.parsed.prompts

            for prompt in prompts:
                csv_writer.writerow([prompt, os.path.basename(code_path)])

            print(f"Generated {len(prompts)} Prompts for {os.path.basename(code_path)}")
            total_prompts += len(prompts)
            successful_files += 1
        
        except FileNotFoundError:
            print(f"Error: File '{code_path}' Not Found")
            failed_files.append(code_path)

        except Exception as e:
            print(f"Error Reading File: {e}")
            failed_files.append(code_path)

print()
print(f"Total Files Processed Successfully: {successful_files}/{len(file_paths)}")
print(f"Total Prompts Generated: {total_prompts}")
print(f"Output Saved to: {os.path.abspath(output_csv)}")

if failed_files:
    print(f"Failed Files ({len(failed_files)}):")
    for failed_file in failed_files:
        print(f"\t - {failed_file}")