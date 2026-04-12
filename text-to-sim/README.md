# PFAGENT Application

This directory contains the main PFAGENT Streamlit application and runtime utilities.

## Entry Point

Run:

```bash
conda activate pfagent
streamlit run main.py
```

## Main Modes

- `Base OpenAI`
- `Fine-tuned`
- `GraphRAG`
- `Fine-tuned + RAG`

Current recommended mode:

- `Fine-tuned + RAG`

## What This App Does

- accepts natural-language power-flow requests
- grounds generation with the ANDES manual
- supports built-in and uploaded ANDES cases
- generates and executes Python code
- captures plots and execution outputs
- supports multi-turn follow-up modifications

## Important Runtime Directories

- `src/`: application modules and chatbot implementations
- `scripts/`: regression and helper scripts
- `tests/`: unit tests
- `code_executions/`: generated at runtime, not source-controlled
- `data_files/`: few-shot data and local runtime metadata

## Notes

- `GraphRAG` requires Neo4j environment variables.
- OpenAI API keys can be provided through the UI.
- Runtime outputs should not be committed back to the repository.

