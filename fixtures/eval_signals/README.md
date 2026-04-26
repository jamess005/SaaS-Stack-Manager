Terminal-only held-out evaluation fixtures live here.

Regenerate them with:

```bash
python training/generate_eval_signals.py
```

Run the evaluator with:

```bash
python scripts/evaluate_model.py --holdout --dry-run
```

Each JSON file keeps metadata such as the expected verdict at the top level and nests the actual signal body under `signal` so the model does not see the answer during evaluation.