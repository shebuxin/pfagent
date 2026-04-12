# Layered domain knowledge distilliation -- high level

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
output_filename = "high_level_generated_prompts.csv"
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
                    You are generating training data for fine-tuning a language model.

                    I will provide you with a piece of code. Your task is to write very high-level user questions that could plausibly lead to this code being generated as an answer.

                    Rules:
                    1. Keep the questions extremely broad and conceptual. Do not mention technical details like active/reactive power, bus voltages, lines, or power flow.
                    - Good: "How can I evaluate the overall condition of the test grid?"
                    - Good: "Summarize the status of the power system."
                    - Bad: "What is the reactive power outputs of PV generators?"
                    2. Focus on overall goals or system-level insights, not calculations or variables.  
                    Use generic words like: *status*, *condition*, *behavior*, *performance*, *overview*, *health*.
                    3. Each question should be short and represent a single intent.  
                    - Good: "Is the network operating normally?"  
                    - Bad: "How can I load a test case and extract all generator outputs and line flows?"
                    4. Imagine the question is asked by a decision-maker or operator who only wants a summary of what the system is doing.
                    5. Output must be a JSON array of short strings, with no commentary.

                                                
                    Generic Examples:

                    Good: “I’d like to know the operational status of the power system (ieee39.xlsx).”
                    Bad: “Generate code that grabs the 39 bus case, runs its analysis, and prints out power figures for PV and slack buses and the voltages and angles for all buses.”
                """).strip()

            # if not example_prompt.empty:
            #     system_prompt = textwrap.dedent(f"""
            #         You are an expert in AI training data generation. Your task is to create natural, clear, varied, and tonally diverse, **extremely high level** questions or instructions that a user might ask to produce **roughly** the given code as the output. The prompts should be high level; thus, be brief and vague. Since the goal is to create natural prompts that a typical user would ask, avoid overly technical or exhaustive specifications (e.g., listing exact parameter values to set to true/false) unless it would be realistic for a normal user. Do not assume the user knows domain-specific technical details (e.g., PQ, PV, Slack bus, or similar concepts); prompts should be written as if the user has little to no awareness of such terms. Think of what someone with little to no technical background might ask. Their questions should sound natural, casual, and high-level — often vague, imprecise, or overlooking important details. Write them as if the person doesn’t know the jargon or inner workings of the system, and is simply describing what they want in plain, everyday language rather than precise technical terms. The less specific and less precise the prompts are, the better. Do not include any labels, annotations, or descriptions before the prompts (e.g., “Prompt: …” or “Short prompt: …”). Respond only with the list of generated prompts. These questions will be used to create high-quality Q&A pairs for fine-tuning a code-generation model.

            #         Example Question Prompt (Should Not Exceed This Example in Terms of Level of Detail): {example_prompt}
            #     """).strip()
            # else:
            #     system_prompt = "You are an expert in AI training data generation. Your task is to create natural, clear, varied, and tonally diverse **extremely high level** questions or instructions that a user might ask to produce **roughly** the given code as the output. The prompts should be high level; thus, be brief and vague. Since the goal is to create natural prompts that a typical user would ask, avoid overly technical or exhaustive specifications (e.g., listing exact parameter values to set to true/false) unless it would be realistic for a normal user. Do not assume the user knows domain-specific technical details (e.g., PQ, PV, Slack bus, or similar concepts); prompts should be written as if the user has little to no awareness of such terms. Think of what someone with little to no technical background might ask. Their questions should sound natural, casual, and high-level — often vague, imprecise, or overlooking important details. Write them as if the person doesn’t know the jargon or inner workings of the system, and is simply describing what they want in plain, everyday language rather than precise technical terms. The less specific and less precise the prompts are, the better. Do not include any labels, annotations, or descriptions before the prompts (e.g., “Prompt: …” or “Short prompt: …”). Respond only with the list of generated prompts. These questions will be used to create high-quality Q&A pairs for fine-tuning a code-generation model."
            #     print("No Example Prompt Found")

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