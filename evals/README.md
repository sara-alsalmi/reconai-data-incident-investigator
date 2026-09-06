# ReconAI TPC-H Evaluation

This evaluation uses the standard TPC-H `orders` table as a realistic wholesale
business schema. The builder creates paired source and warehouse CSVs, injects
controlled incidents, and records hidden ground truth that is never uploaded to
ReconAI.

The deterministic suite covers a clean control, missing orders, duplicated
orders, and amount drift on matching order IDs. The agentic suite adds an
offsetting missing-plus-duplicate incident and a value-drift localization task.

Agentic cases cannot pass from precomputed baseline evidence alone. They require
an Investigator attempt, evidence created by the Investigator's selected tools,
the expected tool path, safe semantics, and Verifier acceptance.

Run the suites separately so deterministic accuracy is not presented as LLM
agent reliability:

```powershell
uv run python -m scripts.run_tpch_evaluation --suite deterministic --runs 3 --output evals/results/deterministic-latest.json
uv run python -m scripts.run_tpch_evaluation --suite agentic --runs 3 --output evals/results/agentic-latest.json
```

Current measured status: the deterministic suite passed 12/12 repeated runs.
The first complete agentic run passed 0/2; follow-up validation exhausted the
configured OpenRouter free tier's daily request allowance. This is a recorded
model limitation, not a passing agent result.

Generated datasets and run results are local artifacts and are not committed.
