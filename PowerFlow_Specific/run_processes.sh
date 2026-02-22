#!/bin/bash

# Sequential execution of Python scripts
# Make sure all required Python files are in the current directory

echo "Starting Sequential Script Execution..."

echo "Step 1: Running high_level_generate_qa_pairs.py"
python high_level_generate_qa_pairs.py
if [ $? -ne 0 ]; then
    echo "Error: high_level_generate_qa_pairs.py Failed"
    exit 1
fi

echo "Step 2: Running low_level_generate_qa_pairs.py"
python low_level_generate_qa_pairs.py
if [ $? -ne 0 ]; then
    echo "Error: low_level_generate_qa_pairs.py Failed"
    exit 1
fi

echo "Step 3: Running summary_generation.py"
python summary_generation.py
if [ $? -ne 0 ]; then
    echo "Error: summary_generation.py Failed"
    exit 1
fi

echo "Step 4: Running fine_tuning_json_generation.py"
python fine_tuning_json_generation.py
if [ $? -ne 0 ]; then
    echo "Error: fine_tuning_json_generation.py Failed"
    exit 1
fi

echo "Step 5: Running finetune.py"
python finetune.py
if [ $? -ne 0 ]; then
    echo "Error: finetune.py Failed"
    exit 1
fi

echo "All Scripts Completed Successfully"