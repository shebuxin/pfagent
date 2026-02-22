import os
import sys
import json
from openai import OpenAI


client = OpenAI(api_key="")
file_folder = "./examples"  
output_summary_file = "combined.jsonl"


def extract_text_and_code(notebook):
    extracted_data = []
    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type")
        if cell_type in ["markdown", "code"]:
            cell_content = "".join(cell.get("source", []))
            extracted_data.append({
                "cell_type": cell_type,
                "content": cell_content.strip()
            })
    return extracted_data



if os.path.exists(output_summary_file) and os.path.getsize(output_summary_file) > 0:
    try:
        with open(output_summary_file, "r") as file:
            combined_data = json.load(file)
        # Ensure combined_data is a list
        if not isinstance(combined_data, list):
            combined_data = [combined_data]
    except json.JSONDecodeError:
        combined_data = []
else:
    combined_data = []


for file_name in os.listdir(file_folder):
    if file_name.lower().endswith(".ipynb"):
        file_name_without_extension, extension = os.path.splitext(file_name)
        if extension: 
            modified_file_name = file_name_without_extension + "_" + extension[1:]  # Remove the dot from extension and append with an underscore
        else:
            modified_file_name = file_name_without_extension  # No extension, so keep the original name

        output_individual_file = "results" + "_" + modified_file_name + ".jsonl"
        file_path = os.path.join(file_folder, file_name)

        with open(file_path, "r") as file:
            notebook = json.load(file)
            # file_content = file.read()

        file_content = extract_text_and_code(notebook)

        # print(file_content)

        messages = [
            {
                "role": "system",
                "content": (
                    """
                    Parse a file containing markdown comments and code snippets to generate training datapoints for model fine-tuning. The goal of these datapoints is to train the LLM to generate code that a human can use. 
                    Adjust your parsing to handle code from both standard Python files and Jupyter Notebooks — note that notebook code may include leading exclamation marks (!) to denote shell commands. 
                    Each datapoint should be in JSON format with the structure: {"messages": [{"role": "user", "content": "Your prompt here"}, {"role": "assistant", "content": "Your corresponding completion here."}]}
                    For each file, generate **multiple** distinct training datapoints. 

                    - **Granularity Variations:**
                    - Break the file into segments focusing on distinct code pieces or aspects.
                    - Create multiple datapoints per file, each addressing a unique aspect or section.
                    - Ensure each training example is self-contained with a clear question, context prompt, request, or command.
                    - Emphasize generating datapoints that improve the LLM's coding capabilities.
                    - The overall intent is to improve the LLM’s capability to generate code that directly meets human instructions.

                    - **Combined Code Data Point:**
                    - Create a final training datapoint that aggregates all non-shell code from the file.
                    - The prompt should be based on the analysis of both the code and the markdown text. 
                    - The prompt should be a command or request, similar in style to: "Generate code for ANDES that performs power flow analysis and plots the results." However, the actual command must accurately reflect the functionality and intent demonstrated by the file’s content. Do not use any static or example text verbatim.
                    - The completion should present the entire code logically and coherently concatenated.
                    - Exclude any shell code (e.g., "!ls").

                    # Steps

                    1. **Parse the File:** Extract markdown comments and code snippets.
                    2. **Generate Segments:**
                    - Identify distinct segments focusing on specific aspects of the code.
                    - For each segment, develop a prompt and a completion that capture the code’s logic and purpose, derived entirely from the code and surrounding commentary.
                    3. **Create Combined Data Point:**
                    - Gather all non-shell code snippets and concatenate them for a comprehensive completion.
                    - Dynamically generate a command-like prompt that reflects the overall purpose and functionality of the combined code. This prompt should reflect what a human would request (e.g., “Generate code for ANDES that performs power flow analysis and plots the results”), tailored to the file's actual content.
                    4. **Format Output:** Ensure all outputs are JSON objects as specified.

                    # Output Format

                    - A list of JSON objects, each representing a training datapoint.
                    - Format each JSON object as: {"messages": [{"role": "user", "content": "Your prompt here"}, {"role": "assistant", "content": "Your corresponding completion here."}]} 
                    """
                )
            },
            {
                "role": "user",
                "content": json.dumps(file_content)  # Convert file_content (list) to a JSON string
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            # max_tokens=300,
        )

        # messages = [
        #     {
        #         "type": "text",
        #         "text": (
        #             """
        #             Parse a file containing markdown comments and code snippets to generate training datapoints for model fine-tuning. Each datapoint should be in JSON format with the structure: {"prompt": "Your prompt here", "completion": "Your corresponding completion here."}

        #             - **Granularity Variations:**
        #             - Break the file into segments focusing on distinct code pieces or aspects.
        #             - Create multiple datapoints per file, each addressing a unique aspect or section.
        #             - Ensure each training example is self-contained with a clear question, context prompt, or explanation in completion.
        #             - Emphasize generating datapoints improving the LLM's coding capabilities.

        #             - **Combined Code Data Point:**
        #             - Create a final training datapoint aggregating all the file's code.
        #             - The completion should present the entire code logically and coherently concatenated.

        #             # Steps

        #             1. **Parse the File:** Extract markdown comments and code snippets from the file.
        #             2. **Generate Segments:**
        #             - Identify distinct segments focusing on specific aspects of the code.
        #             - Develop prompts and completions for each segment to form self-contained training examples.
        #             3. **Create Combined Data Point:**
        #             - Gather all code snippets and concatenate them into a comprehensive completion.
        #             4. **Format Output:** Ensure all outputs are JSON objects as specified.

        #             # Output Format

        #             - A list of JSON objects, each representing a training data point.
        #             - Format each JSON object as: `{"prompt": "Your prompt here", "completion": "Your corresponding completion here."}`
        #             """

        #             # """
        #             # You are provided with a file that contains various markdown comments and code snippets. Your task is to generate multiple training datapoints for model fine tuning. Each training data point must be in JSON format with the following structure:

        #             # {"prompt": "Your prompt here", "completion": " Your corresponding completion here."}

                    
        #             # Follow these instructions:

        #             # 1, Granularity Variations:

        #             #     * Parse the file and break it down into multiple segments. For each segment, generate a separate training example that focuses on a specific piece or aspect of the code. There should be multiple datapoints per file, each focusing on a different aspect or section.
        #             #     * Ensure that each training example is self-contained. The prompt should ask a question or provide a context derived from the file, and the completion should offer a clear and concise answer, explanation, or code based on that segment. 
        #             #     * Focus mostly on generate datapoints that helps the LLM become a better coder.

        #             # 2, Combined Code Data Point:

        #             #     * In addition to the segmented examples, create one final training data point that aggregates all the code from the file. For this combined data point:
        #             #         ** The completion should consist of the entire code from the file concatenated in a logical and coherent manner.
                    
        #             # 3, Output Format:

        #             #     * Important: Do not wrap your output in any additional keys or objects (e.g., do not use {"training_data": [...]}).
        #             #     * Each training data point must strictly follow the JSON structure shown above.

        #             # Your output should be a list of JSON objects where each object represents one training data point. This training data will then be used for fine-tuning the model.
        #             # """

        #             # "You are a notebook-to-JSON conversion assistant. Your task is to extract training examples from a Jupyter Notebook and output them as a JSON array. For each code cell, do the following:"
        #             # "- If one or more markdown cells immediately precede the code cell, concatenate their text to form the \"prompt\"."
        #             # "- Use the code cell’s content as the \"completion\"."
        #             # "- If a code cell is not preceded by any markdown, set the \"prompt\" to an empty string."
        #             # "- Do not include any extra commentary or markdown formatting; output only valid JSON."
        #             # "Create a JSON file from the contents of a Jupyter Notebook for fine-tuning purposes, ensuring to add data of various granularity."
        #             # "Use the data within the Jupyter Notebook to extract relevant sections, including code, markdown, and any outputs. Ensure you structure the JSON to capture different levels of detail for fine-tuning purposes, including high-level summaries, detailed sections, and potentially individual outputs where beneficial."
        #         )
        #     },
        #     {
        #         "type": "text",
        #         "text": json.dumps(file_content)  # Convert list to a JSON string
        #     }
        # ]
        
        # response = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[{"role": "user", "content": messages}],
        #     response_format={"type": "json_object"},
        #     # max_tokens=300,
        # )
        
        output = response.choices[0].message.content

        # print(output)
        
        try:
            result_json = json.loads(output)
            # result_json = json.loads(output.model_dump_json())
        except Exception as e:
            print(f"Error: Could not parse output: {output}. Exception: {e}")
            continue
            # sys.exit(1)

        if isinstance(result_json, dict):
            if not (len(result_json) == 1 and "messages" in result_json):
                combined_items = []
                for key, value in result_json.items():
                    if isinstance(value, list):
                        combined_items.extend(value)
                    else:
                        combined_items.append(value)
                result_json = combined_items

        # if isinstance(result_json, dict):
        #     combined_items = []
        #     for key, value in result_json.items():
        #         if isinstance(value, list):
        #             combined_items.extend(value)
        #         else:
        #             combined_items.append(value)
        #     result_json = combined_items

        # if isinstance(result_json, dict) and len(result_json) == 1:
        #     result_json = next(iter(result_json.values()))
        
        combined_data.append(result_json)

        with open(output_individual_file, "w") as individual_file:
            if isinstance(result_json, list):
                for item in result_json:
                    individual_file.write(json.dumps(item) + "\n")
            else:
                individual_file.write(json.dumps(result_json) + "\n")

        # with open(output_individual_file, "w") as individual_file:
        #     json.dump(result_json, individual_file)

with open(output_summary_file, "w") as combined_file:
    for dataset in combined_data:
        if isinstance(dataset, list):
            for item in dataset:
                combined_file.write(json.dumps(item) + "\n")
        else:
            combined_file.write(json.dumps(dataset) + "\n")

# with open(output_summary_file, "w") as combined_file:
#     for item in combined_data:
#         combined_file.write(json.dumps(item) + "\n")

# with open(output_summary_file, "w") as combined_file:
#     json.dump(combined_data, combined_file)