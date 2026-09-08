# ReconAI TPC-H Evaluation

This evaluation uses the standard TPC-H `orders` table as a realistic wholesale
business schema. The builder creates paired source and warehouse CSVs, injects
controlled incidents, and records hidden ground truth that is never uploaded to
ReconAI.

The baseline suite (kept as `--suite deterministic` for CLI compatibility) covers a clean
control, missing orders, duplicated orders, and amount drift on matching order IDs. The adaptive
suite (`--suite agentic`) adds an offsetting missing-plus-duplicate incident and a value-drift
localization task. Both suites always run the Investigator and Verifier; the distinction is
whether new post-baseline tool calls are required.
The offsetting case requires both component totals, the true extra-copy count,
an arithmetic closure check, affected-key previews, and a non-contradictory report.

Adaptive cases require an Investigator attempt, fresh post-baseline deterministic
evidence, the expected analysis path, safe semantics, and Verifier/Controller
acceptance. The checks infer relevant keys, measures, and grouping columns from
the question and schema; they do not hardcode TPC-H field values or answers.

Run the suites separately to distinguish baseline reuse from adaptive tool selection:

```powershell
uv run --group eval python -m scripts.build_tpch_evaluation
uv run python -m scripts.run_tpch_evaluation --suite deterministic --runs 3 --output evals/results/deterministic-latest.json
uv run python -m scripts.run_tpch_evaluation --suite agentic --runs 3 --output evals/results/agentic-latest.json
```

Current measured status: **51/51 automated tests** and **2/2 cases in the latest single-run
adaptive TPC-H checks**. The adaptive result confirms both scenarios work end to end, but one
run is not a statistical
reliability claim. Free-model output quality and provider availability can vary,
so use `--runs 3` or more when comparing models or preparing production evidence.

Generated datasets and run results are local artifacts and are not committed.
