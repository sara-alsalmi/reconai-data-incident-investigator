"""Run ReconAI repeatedly and score tool evidence against hidden TPC-H ground truth."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.flow import ReconAIInvestigationFlow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "data" / "tpch-evaluation"
DEFAULT_OUTPUT = PROJECT_ROOT / "evals" / "results" / "latest.json"


def _matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and isinstance(value, (int, float)):
        return abs(float(value) - float(expected)) <= 0.01
    return value == expected


def score_run(result: Any, expected: dict) -> dict:
    checks = []
    for requirement in expected.get("checks", []):
        matches = []
        for evidence in result.state.evidence:
            details = evidence.supporting_details
            if evidence.tool != requirement["tool"]:
                continue
            if requirement.get("dataset") and details.get("dataset") != requirement["dataset"]:
                continue
            if requirement.get("dataset_a") and details.get("dataset_a") != requirement["dataset_a"]:
                continue
            if evidence.attempt < requirement.get("attempt_min", 0):
                continue
            if any(
                not _matches(details.get(key), value)
                for key, value in requirement.get("details", {}).items()
            ):
                continue
            if _matches(details.get(requirement["field"]), requirement["value"]):
                matches.append(evidence.evidence_id)
        checks.append(
            {
                "tool": requirement["tool"],
                "field": requirement["field"],
                "expected": requirement["value"],
                "passed": bool(matches),
                "evidence_ids": matches,
            }
        )
    hypothesis_text = (result.state.hypothesis or "").lower()
    text = (
        hypothesis_text
        if expected.get("semantic_scope") == "hypothesis"
        else f"{result.state.hypothesis or ''}\n{result.report}".lower()
    )
    terms = expected.get("required_terms", [])
    all_terms = expected.get("required_all_terms", [])
    term_groups = expected.get("required_term_groups", [])
    semantic_pass = (
        (any(term.lower() in text for term in terms) if terms else True)
        and all(term.lower() in text for term in all_terms)
        and all(
            any(term.lower() in text for term in group) for group in term_groups
        )
    )
    hypothesis = hypothesis_text
    forbidden_terms = expected.get("forbidden_hypothesis_terms", [])
    causal_safety_pass = not any(
        term.lower() in hypothesis for term in forbidden_terms
    )
    evidence_pass = all(item["passed"] for item in checks)
    attempts = result.state.attempt_count
    agent_evidence = [item for item in result.state.evidence if item.attempt > 0]
    min_attempts = int(expected.get("min_attempts", 0))
    min_agent_tool_calls = int(expected.get("min_agent_tool_calls", 0))
    required_agent_tools = expected.get("required_agent_tools", {})
    agent_tool_counts = {
        tool: sum(1 for item in agent_evidence if item.tool == tool)
        for tool in required_agent_tools
    }
    observed_verdict = (
        result.state.verification.verdict.value if result.state.verification else None
    )
    required_verdict = expected.get("required_verdict")
    verdict_pass = required_verdict is None or observed_verdict == required_verdict
    agentic_pass = (
        attempts >= min_attempts
        and len(agent_evidence) >= min_agent_tool_calls
        and all(
            agent_tool_counts.get(tool, 0) >= minimum
            for tool, minimum in required_agent_tools.items()
        )
        and verdict_pass
    )
    return {
        "passed": evidence_pass and semantic_pass and causal_safety_pass and agentic_pass,
        "evidence_checks_passed": evidence_pass,
        "semantic_check_passed": semantic_pass,
        "causal_safety_passed": causal_safety_pass,
        "agentic_requirements_passed": agentic_pass,
        "attempts_observed": attempts,
        "agent_tool_calls_observed": len(agent_evidence),
        "agent_tool_counts": agent_tool_counts,
        "verdict_requirement_passed": verdict_pass,
        "verdict_observed": observed_verdict,
        "forbidden_hypothesis_terms_found": [
            term for term in forbidden_terms if term.lower() in hypothesis
        ],
        "checks": checks,
    }


def run(
    cases_root: Path,
    output: Path,
    runs: int,
    selected_cases: list[str],
    suite: str = "all",
) -> dict:
    case_dirs = sorted(path for path in cases_root.iterdir() if path.is_dir())
    if selected_cases:
        wanted = set(selected_cases)
        case_dirs = [path for path in case_dirs if path.name in wanted]
        missing = wanted - {path.name for path in case_dirs}
        if missing:
            raise ValueError(f"Unknown case(s): {', '.join(sorted(missing))}")
    if suite != "all":
        require_agent = suite == "agentic"
        case_dirs = [
            path
            for path in case_dirs
            if bool(
                json.loads((path / "ground_truth.json").read_text(encoding="utf-8"))[
                    "expected"
                ].get("requires_agent", False)
            )
            == require_agent
        ]
    records = []
    for case_dir in case_dirs:
        manifest = json.loads((case_dir / "ground_truth.json").read_text(encoding="utf-8"))
        datasets = [str(case_dir / name) for name in manifest["datasets"]]
        for run_number in range(1, runs + 1):
            print(f"Running {case_dir.name} ({run_number}/{runs})", flush=True)
            try:
                result = ReconAIInvestigationFlow(
                    manifest["question"], datasets
                ).run()
                score = score_run(result, manifest["expected"])
                records.append(
                    {
                        "case_id": case_dir.name,
                        "requires_agent": bool(
                            manifest["expected"].get("requires_agent", False)
                        ),
                        "run": run_number,
                        "passed": score["passed"],
                        "score": score,
                        "attempts": result.state.attempt_count,
                        "verdict": result.state.verification.verdict.value
                        if result.state.verification
                        else None,
                        "hypothesis": result.state.hypothesis,
                        "evidence": [
                            item.model_dump(mode="json") for item in result.state.evidence
                        ],
                        "report": result.report,
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "case_id": case_dir.name,
                        "run": run_number,
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    passed = sum(1 for record in records if record["passed"])
    suite_results = {}
    for name, requires_agent in (("deterministic", False), ("agentic", True)):
        matching = [
            record
            for record in records
            if record.get("requires_agent", False) == requires_agent
        ]
        matching_passed = sum(1 for record in matching if record["passed"])
        suite_results[name] = {
            "runs": len(matching),
            "passed": matching_passed,
            "pass_rate": 0.0
            if not matching
            else round(matching_passed / len(matching) * 100, 2),
        }
    summary = {
        "benchmark": "TPC-H reconciliation incidents",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": len(records),
        "passed": passed,
        "pass_rate": 0.0 if not records else round(passed / len(records) * 100, 2),
        "suites": suite_results,
        "results": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Passed {passed}/{len(records)} runs ({summary['pass_rate']:.2f}%)")
    print(f"Detailed results: {output}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--suite", choices=("all", "deterministic", "agentic"), default="all"
    )
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    run(args.cases_root, args.output, args.runs, args.case, args.suite)


if __name__ == "__main__":
    main()
