import html
import json
import os
from pathlib import Path

import gradio as gr

BUNDLE_PATH = Path(os.environ.get("GRPO_BUNDLE", Path(__file__).parent / "bundle" / "examples.json"))
with open(BUNDLE_PATH) as f:
    EXAMPLES = json.load(f)

EXAMPLE_CHOICES = [ex["title"] for ex in EXAMPLES.values()]
TITLE_TO_ID = {ex["title"]: k for k, ex in EXAMPLES.items()}

CSS = """
.model-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 14px;
    background: #fafafa;
    margin-bottom: 12px;
}
.badge-pass {
    background: #e6f4ea;
    color: #137333;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    display: inline-block;
    margin-right: 6px;
}
.badge-fail {
    background: #fce8e6;
    color: #c5221f;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    display: inline-block;
    margin-right: 6px;
}
.model-title {
    font-size: 14px;
    font-weight: bold;
    color: #202124;
    margin-bottom: 6px;
}
.model-desc {
    font-size: 11px;
    color: #5f6368;
    margin-bottom: 8px;
}
.gen-text {
    font-family: monospace;
    font-size: 12px;
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 6px;
    padding: 10px;
    white-space: pre-wrap;
    max-height: 220px;
    overflow-y: auto;
}
"""

MODEL_METAS = {
    "base": {
        "title": "1. Base Model (No RL Training)",
        "desc": "Untrained Qwen2.5-0.5B-Instruct baseline with 8-shot chain-of-thought prompt."
    },
    "strict": {
        "title": "2. Trained on Strict Grader (+1.0 for valid <answer> tag & correct answer)",
        "desc": "Learns structured multi-step reasoning and maintains strict <answer>X</answer> tag discipline."
    },
    "lenient_last": {
        "title": "3. Trained on Last Number (+1.0 if last number matches)",
        "desc": "Failure Mode 1: Format Collapse. Solves the arithmetic, but abandons the <answer> tag entirely."
    },
    "any_match": {
        "title": "4. Trained on Any Match (+1.0 if any number matches)",
        "desc": "Failure Mode 2: Reward Hacking. Rambles and enumerates numbers to hit the target by coincidence."
    },
    "format_bonus": {
        "title": "5. Trained on Format Bonus (+0.2 for valid tag + 1.0 for correct)",
        "desc": "Failure Mode 3: Length Collapse. Collapses to ~9 tokens, skipping reasoning to cash the format reward."
    }
}

def render_card(key, data):
    meta = MODEL_METAS[key]
    badges = []
    for rubric, name in [("strict", "Strict Grader"), ("lenient_last", "Last Number"), ("any_match", "Any Match")]:
        status = data.get(rubric, False)
        cls = "badge-pass" if status else "badge-fail"
        label = f"{name}: PASS" if status else f"{name}: FAIL"
        badges.append(f'<span class="{cls}">{label}</span>')
    badges_html = " ".join(badges)
    text_escaped = html.escape(data.get("text", ""))
    
    return f"""
    <div class="model-card">
        <div class="model-title">{meta['title']}</div>
        <div class="model-desc">{meta['desc']}</div>
        <div style="margin-bottom: 8px;">{badges_html}</div>
        <div class="gen-text">{text_escaped}</div>
    </div>
    """

def on_select_example(choice):
    qid = TITLE_TO_ID.get(choice, list(EXAMPLES.keys())[0])
    ex = EXAMPLES[qid]
    
    cards_html = "".join([render_card(m, ex["models"][m]) for m in ["base", "strict", "lenient_last", "any_match", "format_bonus"]])
    return ex["question"], ex["answer"], cards_html

def build_app():
    with gr.Blocks(title="GRPO Grader Leniency Demo", css=CSS) as demo:
        gr.Markdown(
            "# GRPO From Scratch: What a Lenient Grader Teaches a Language Model\n"
            "Interactive demonstration comparing **Qwen2.5-0.5B-Instruct** trained under four distinct RL grader reward functions on GSM8K.\n\n"
            "Notice how subtle leniencies produce **format collapse**, **reward hacking**, or **length collapse**."
        )
        with gr.Row():
            with gr.Column(scale=1):
                example_dropdown = gr.Dropdown(
                    choices=EXAMPLE_CHOICES,
                    value=EXAMPLE_CHOICES[0],
                    label="Select a Benchmark Example"
                )
                question_box = gr.Textbox(
                    label="Math Question",
                    lines=4,
                    interactive=False
                )
                answer_box = gr.Number(
                    label="Expected Numeric Answer",
                    interactive=False
                )
            with gr.Column(scale=2):
                results_view = gr.HTML()
                
        example_dropdown.change(
            on_select_example,
            inputs=[example_dropdown],
            outputs=[question_box, answer_box, results_view]
        )
        
        # Load initial example on launch
        demo.load(
            on_select_example,
            inputs=[example_dropdown],
            outputs=[question_box, answer_box, results_view]
        )
    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7861)
