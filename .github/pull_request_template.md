## Summary

Describe the change at a high level.

## Area

- [ ] text-to-sim app
- [ ] fine-tuning pipeline
- [ ] verification suite
- [ ] docs / repository standards

## Validation

- [ ] `python -m unittest discover -s text-to-sim/tests -p "test_*.py"`
- [ ] `python -m unittest discover -s verification/tests -p "test_*.py"`
- [ ] `python knowledge/finetuning/scripts/smoke_test_finetune_examples.py`
- [ ] Additional validation was run when needed

## Data Governance Check

- [ ] No accidental holdout-to-training leakage was introduced
- [ ] No generated runtime artifacts were committed unintentionally

## Notes

Anything reviewers should pay special attention to.
