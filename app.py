"""Minimal Gradio UI for ReconAI."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from src.config import ConfigurationError, friendly_llm_error
from src.flow import ReconAIInvestigationFlow
from src.tools.common import ToolInputError


def investigate(files: list[str] | None, question: str) -> tuple[str, str]:
    paths = [str(Path(item)) for item in (files or [])]
    try:
        result = ReconAIInvestigationFlow(question, paths).run()
        return result.report, result.trace
    except (ConfigurationError, ToolInputError) as exc:
        return f"## Unable to investigate\n\n{exc}", "Investigation did not start."
    except Exception as exc:  # provider SDKs use several exception types
        return f"## Unable to investigate\n\n{friendly_llm_error(exc)}", (
            "The run stopped safely. No hidden reasoning or secrets were exposed."
        )


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="ReconAI") as demo:
        gr.Markdown(
            "# ReconAI — Multi-Agent Data Incident Investigator\n"
            "Upload CSV datasets and ask a reconciliation or incident question."
        )
        uploads = gr.File(
            label="Dataset Upload",
            file_count="multiple",
            file_types=[".csv"],
            type="filepath",
        )
        question = gr.Textbox(
            label="Investigation Question",
            placeholder="Why does revenue in payments.csv not match orders.csv?",
            lines=2,
        )
        button = gr.Button("Investigate", variant="primary")
        report = gr.Markdown(label="Investigation Report")
        trace = gr.Textbox(label="Investigation Trace", lines=14, interactive=False)
        button.click(investigate, inputs=[uploads, question], outputs=[report, trace])
    return demo


if __name__ == "__main__":
    build_interface().queue().launch()
