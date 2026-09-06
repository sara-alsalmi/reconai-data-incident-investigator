"""Minimal Gradio UI for ReconAI."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from src.config import ConfigurationError, friendly_llm_error
from src.flow import ReconAIInvestigationFlow
from src.tools.common import ToolInputError


USAGE_GUIDE = """
### Run an investigation in three steps

1. **Upload related CSV files** that share an identifier such as `order_id`, `invoice_id`,
   `transaction_id`, or `customer_id`.
2. **Ask a normal question.** You do not need to explain the schema or choose tools.
3. **Select Investigate** and review the evidence-backed report.

Try one of these questions:

- `Investigate why these datasets disagree and show the affected records.`
- `Which IDs have no corresponding record in the other file?`
- `Which field or business segment explains the difference?`

ReconAI automatically profiles the files, infers the shared-key relationship, checks both
directions, and distinguishes exact duplicates from valid one-to-many records. A high-confidence
data finding can still have an inconclusive technical root cause because CSV files do not contain
application or pipeline logs.
"""


RECONAI_THEME = gr.themes.Default(
    primary_hue="slate",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="#f3f4f6",
    body_background_fill_dark="#111827",
    body_text_color="#111827",
    body_text_color_dark="#f3f4f6",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#1f2937",
    block_background_fill="#ffffff",
    block_background_fill_dark="#1f2937",
    block_border_color="#d1d5db",
    block_border_color_dark="#374151",
    input_background_fill="#f9fafb",
    input_background_fill_dark="#111827",
    input_border_color_focus="#111827",
    input_border_color_focus_dark="#9ca3af",
    button_primary_background_fill="#111827",
    button_primary_background_fill_hover="#1f2937",
    button_primary_background_fill_dark="#374151",
    button_primary_background_fill_hover_dark="#4b5563",
    button_primary_border_color="#111827",
    button_primary_border_color_dark="#4b5563",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    color_accent_soft="#e5e7eb",
    color_accent_soft_dark="#374151",
    link_text_color="#374151",
    link_text_color_dark="#d1d5db",
)


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
            "Compare related CSV datasets and receive a clear, evidence-checked report."
        )
        with gr.Accordion("How to use ReconAI", open=False):
            gr.Markdown(USAGE_GUIDE)
        uploads = gr.File(
            label="1. Upload related CSV files",
            file_count="multiple",
            file_types=[".csv"],
            type="filepath",
        )
        question = gr.Textbox(
            label="2. Ask an investigation question",
            placeholder=(
                "Example: Investigate why these datasets disagree and show the "
                "affected records."
            ),
            lines=2,
        )
        button = gr.Button("Investigate", variant="primary")
        report = gr.Markdown(label="Investigation Report")
        trace = gr.Textbox(
            label="Investigation Trace — Agent Processing",
            lines=14,
            interactive=False,
        )
        button.click(investigate, inputs=[uploads, question], outputs=[report, trace])
    return demo


if __name__ == "__main__":
    build_interface().queue().launch(theme=RECONAI_THEME)
