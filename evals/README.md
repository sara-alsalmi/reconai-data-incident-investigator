# ReconAI TPC-H Evaluation

This evaluation uses the standard TPC-H `orders` table as a realistic wholesale
business schema. The builder creates paired source and warehouse CSVs, injects
controlled incidents, and records hidden ground truth that is never uploaded to
ReconAI.

The deterministic suite covers a clean control, missing orders, duplicated
orders, and amount drift on matching order IDs. The agentic suite adds an
offsetting missing-plus-duplicate incident and a value-drift localization task.

Agentic cases require an Investigator attempt, fresh post-baseline deterministic
evidence, the expected analysis path, safe semantics, and Verifier/Controller
acceptance. The checks infer relevant keys, measures, and grouping columns from
the question and schema; they do not hardcode TPC-H field values or answers.

Run the suites separately so deterministic accuracy is not presented as LLM
agent reliability:

```powershell
uv run --group eval python -m scripts.build_tpch_evaluation
uv run python -m scripts.run_tpch_evaluation --suite deterministic --runs 3 --output evals/results/deterministic-latest.json
uv run python -m scripts.run_tpch_evaluation --suite agentic --runs 3 --output evals/results/agentic-latest.json
```

Current measured status: the deterministic suite passed **12/12 repeated runs**
and the latest single-run agentic suite passed **2/2 cases**. The agentic result
confirms both scenarios work end to end, but one run is not a statistical
reliability claim. Free-model output quality and provider availability can vary,
so use `--runs 3` or more when comparing models or preparing production evidence.

Generated datasets and run results are local artifacts and are not committed.
