CONDA_ENV ?= pfagent
PYTHON ?= conda run -n $(CONDA_ENV) python

# Override CONDA_ENV from the command line when the local env has a different
# name, e.g.:  CONDA_ENV=poweragent make test

.PHONY: app test test-app test-verification smoke-finetune verify fine-tune-data clean-generated \
        snapshot-check snapshot-update

app:
	cd text-to-sim && streamlit run main.py

test: test-app test-verification

test-app:
	$(PYTHON) -m unittest discover -s text-to-sim/tests -p "test_*.py"

test-verification:
	$(PYTHON) -m unittest discover -s verification/tests -p "test_*.py"

# Snapshot guard for the upcoming rag_chatbot.py refactor (see Stage 0 plan).
# snapshot-check runs the pinned-output tests in compare mode (CI-safe).
# snapshot-update regenerates the JSON baseline; only use after a deliberate
# behavioral change that has been reviewed.
snapshot-check:
	$(PYTHON) -m unittest text-to-sim.tests.test_rag_chatbot_snapshots

snapshot-update:
	UPDATE_SNAPSHOTS=1 $(PYTHON) -m unittest text-to-sim.tests.test_rag_chatbot_snapshots

smoke-finetune:
	$(PYTHON) knowledge/finetuning/scripts/smoke_test_finetune_examples.py

verify:
	$(PYTHON) verification/runner.py --scenario-count 100

fine-tune-data:
	$(PYTHON) knowledge/finetuning/scripts/run_processes.py

clean-generated:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf text-to-sim/code_executions text-to-sim/regression_runtime
	rm -rf verification/artifacts verification/artifacts_parallel
	rm -rf verification/artifacts_optimized verification/artifacts_ft_compare_old
	rm -rf verification/artifacts_ft_compare_new verification/artifacts_ft_compare_comparison
