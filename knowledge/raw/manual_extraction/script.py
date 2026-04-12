from pydantic import BaseModel
from typing import List
import fitz  # pip install pymupdf
import re

import json
import os
import csv

from openai import OpenAI


#########################################################################################
# PDF Processing
#########################################################################################


class PageRanges(BaseModel):
    page_ranges: List[List[int]]


def get_toc(pdf_path):
    doc = fitz.open(pdf_path)
    raw_toc = doc.get_toc()
    return [(section_title, page_number) for heading_level, section_title, page_number in raw_toc]


def extract_logical_page_range(pdf_path, start, end):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(start-1, end):
        page = doc.load_page(i)
        pages.append(page.get_text("text")) 
    doc.close()
    return "\n\n".join(pages)


def iter_subsections(text, header_pattern = r'^(?P<header>\d+(?:\.\d+)*\s+.+)$', ignore_patterns = None):
    header_regex = re.compile(header_pattern)
    ignore_regex = [re.compile(pattern) for pattern in (ignore_patterns or [])]

    current_header = None
    buffer = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if any(regex.match(stripped) for regex in ignore_regex):
            continue

        matched_header = header_regex.match(stripped)
        if matched_header:
            if current_header is not None:
                yield current_header, "\n".join(buffer).strip()
            current_header = matched_header.group("header")
            buffer = []
        elif current_header is not None:
            buffer.append(raw_line)

    if current_header is not None:
        yield current_header, "\n".join(buffer).strip()


#########################################################################################
# Get APIs
#########################################################################################


def get_api_page_ranges(pdf_file):
    toc_list = get_toc(pdf_file)

    client = OpenAI()

    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {"role": "system", "content": '''
                You have access to the table of contents of a reference manual. Scan the table of contents to locate every section that defines one or more API functions. Then output **only** a Python list of `(start_page, end_page)` tuples indicating the inclusive page-range for each such section.
            '''},
            {"role": "user", "content": str(toc_list)},
        ],
        text_format=PageRanges,
    )

    return response.output_parsed.page_ranges


def add_apis_to_csv(functions):
    """
    Append rows to a CSV file with columns:
      Function Description | Function Name | Function Parameters

    `functions` should be a list of dicts, each with keys:
      - description: str
      - name: str
      - parameters: List[str]
    """

    csv_file = "ANDES_APIs.csv"
    write_header = not os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Function Description", "Function Name", "Function Parameters"])

        for function in functions:
            description = function["description"]
            name = function["name"]
            parameters = function["parameters"]
            parameters_string = ",".join(parameters)
            writer.writerow([description, name, parameters_string])


def extract_APIs(text):
    client = OpenAI()

    tools = [{
        "type": "function",
        "name": "add_apis_to_csv",
        "description": "Append one or more rows to a CSV file",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "functions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": { "type": "string" },
                            "name":        { "type": "string" },
                            "parameters": {
                                "type":  "array",
                                "items": { "type": "string" }
                            }
                        },
                        "required": ["description", "name", "parameters"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["functions"],
            "additionalProperties": False
        }
    }]

    response = client.responses.create(
        model="o4-mini-2025-04-16",
        reasoning={"effort": "high"},
        input=[
            {"role": "system", "content": '''
                You are an automated extractor that reads a section of a library manual and emits a single call to the `add_apis_to_csv` tool, capturing every API definition found.

                Requirements:
                1. Use exactly one invocation of `add_apis_to_csv(...)`.
                2. Your argument must be a JSON object with a single key `functions`,
                   whose value is a list of objects each having:
                     - description: plain-language summary
                     - name: exact symbol or method name
                     - parameters: list of parameter signatures
                3. Do not emit any other text or code—only the `add_apis_to_csv` call.
            '''},
            {"role": "user", "content": text}
        ],
        tools=tools
    )

    tool_call = next(
        (item for item in response.output
            if getattr(item, "type", None) == "function_call"),
        None
    )

    if tool_call:
        args = json.loads(tool_call.arguments)
        functions = args["functions"]
        add_apis_to_csv(functions)


#########################################################################################
# Get Example Code
#########################################################################################


class ExampleEntry(BaseModel):
    description: str
    prompts: List[str]
    code: str


class ExamplesOutput(BaseModel):
    functions: List[ExampleEntry]


def get_example_page_ranges(pdf_file):
    toc_list = get_toc(pdf_file)

    client = OpenAI()

    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {"role": "system", "content": '''
                You have access to the table of contents of a reference manual. Using the table of contents, your goal is to locate every section that may contain example code snippets and report their inclusive page ranges. Output **only** a Python list of `(start_page, end_page)` tuples indicating the inclusive page-range for each such section.
            '''},
            {"role": "user", "content": str(toc_list)},
        ],
        text_format=PageRanges,
    )

    return response.output_parsed.page_ranges


def add_examples_to_csv(examples):
    """
    Append rows to a CSV file with columns:
      Code Description | Prompt Examples | Code Snippet

    `examples` should be a list of dicts, each with keys:
      - description: str
      - prompts: List[str]
      - code: str
    """

    csv_file = "ANDES_Examples.csv"
    write_header = not os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Code Description", "Prompt Examples", "Code Snippet"])

        for example in examples:
            description = example["description"]
            prompts = example["prompts"]
            prompts_string = ",".join(prompts)
            code = example["code"]
            writer.writerow([description, prompts_string, code])


def extract_examples(text):
    client = OpenAI()

    tools = [{
        "type": "function",
        "name": "add_examples_to_csv",
        "description": "Append one or more rows to a CSV file",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "functions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": { "type": "string" },
                            "prompts": {
                                "type":  "array",
                                "items": { "type": "string" }
                            },
                            "code":        { "type": "string" }
                        },
                        "required": ["description", "prompts", "code"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["functions"],
            "additionalProperties": False
        }
    }]


    response = client.responses.parse(
        model="o4-mini-2025-04-16",
        reasoning={"effort": "high"},
        input=[
            {"role": "system", "content": '''
                You are an automated extractor that reads a section of a library manual and emits a single call to the `add_examples_to_csv` tool, capturing every example code snippet found.

                Requirements:
                1. Emit exactly one invocation of `add_examples_to_csv(...)`.
                2. Its sole argument is an object with a single key, `"functions"`, whose value is a list of objects.
                3. Each object in the list must include:
                - `description` (string): a brief summary in plain English.
                - `prompts` (array of strings): possible user prompts that would elicit this code.
                - `code` (string): the extracted code snippet.
                4. Do not output anything else—no explanations, no formatting, just the function call.
            '''},
            {"role": "user", "content": text}
        ],
        tools=tools
    )

    tool_call = next(
        (item for item in response.output
            if getattr(item, "type", None) == "function_call"),
        None
    )

    if tool_call:
        # args = json.loads(tool_call.arguments)
        # functions = args["functions"]
        # add_examples_to_csv(functions)
        parsed = ExamplesOutput.model_validate_json(tool_call.arguments)
        add_examples_to_csv([entry.model_dump() for entry in parsed.functions])


#########################################################################################
# Main
#########################################################################################


def main():
    API_range_list = [[815, 915]]
    examples_range_list = [[147, 201], [212, 246], [258, 272], [417, 424]]

    pdf_file = "docs-andes-app-en-stable.pdf"

    IGNORE = [
        r'^ANDES Manual, Release',  # document title
        r'^\d{1,3}$',               # standalone page numbers
    ]

    ###

    if API_range_list is None:
        API_range_list = get_api_page_ranges(pdf_file)
        
    print(API_range_list)

    for start, end in API_range_list:
        text = extract_logical_page_range(pdf_file, start, end)
        for header, body in iter_subsections(text, ignore_patterns=IGNORE):
            extract_APIs(header + "\n" + body)

    ###

    if examples_range_list is None:
        examples_range_list = get_example_page_ranges(pdf_file)
        
    print(examples_range_list)

    for start, end in examples_range_list:
        text = extract_logical_page_range(pdf_file, start, end)
        for header, body in iter_subsections(text, ignore_patterns=IGNORE):
            extract_examples(header + "\n" + body)


if __name__ == "__main__":
    main()