"""Interactive Gradio interface for manually testing Model H (Hindi) and
Model L (Nepali) -- pretrained or finetuned -- side by side. This is a
convenience/demo tool (e.g. for a viva), not a graded evaluation script;
the actual metrics live in run_intrinsic.py / run_generation.py /
run_reasoning.py / run_attention.py.

Requires `gradio` (not in requirements.txt, since this tool isn't part of
the graded pipeline): `pip install gradio`

Colab:
    !pip install -q gradio
    !python -m common.eval.interact \\
        --hindi_pretrained  /content/drive/MyDrive/LMA/hindi/output/best.pt \\
        --hindi_finetuned   /content/drive/MyDrive/LMA/hindi/finetune_output/best.pt \\
        --nepali_pretrained /content/drive/MyDrive/LMA/nepali/output/best.pt \\
        --nepali_finetuned  /content/drive/MyDrive/LMA/nepali/finetune_output/best.pt \\
        --share
(--share prints a public gradio.live link since Colab has no local browser
to open a plain http://127.0.0.1 URL against.)

Local (after downloading checkpoints from the Drive links in README.md):
    pip install gradio
    python -m common.eval.interact --hindi_pretrained hindi/model/best.pt ...

Any of the four --*_pretrained/--*_finetuned flags may be omitted if you
only want to test one language or one checkpoint variant -- that
combination simply won't be selectable in the UI.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import sentencepiece as spm
import torch

from common.eval.generate import generate
from common.model.config import GPTConfig
from common.model.transformer import GPT
from common.train.checkpoint import load_checkpoint

LANGS = {
    "hindi": {
        "dir": REPO_ROOT / "hindi",
        "label": "Model H (Hindi)",
        "spm": REPO_ROOT / "hindi" / "tokenizer" / "vocab" / "hindi_spm.model",
        "cfg": REPO_ROOT / "hindi" / "configs" / "model_config.yaml",
    },
    "nepali": {
        "dir": REPO_ROOT / "nepali",
        "label": "Model L (Nepali)",
        "spm": REPO_ROOT / "nepali" / "tokenizer" / "vocab" / "nepali_spm.model",
        "cfg": REPO_ROOT / "nepali" / "configs" / "model_config.yaml",
    },
}

EXAMPLES = [
    # (lang, prompt, max_new_tokens, temperature) -- variant is left as
    # whatever's currently selected, since a reasoning prompt like these is
    # meant to be tried on *both* the pretrained and finetuned checkpoint
    # for comparison (pretrained should fail to produce a bare entity name;
    # finetuned should succeed -- see report/phase3/report.md).
    ["hindi", "पता है कि राम की ऊँचाई, विकास की ऊँचाई से कम है, और विकास की ऊँचाई, सीता की ऊँचाई से कम है। इनमें सबसे कम ऊँचाई किसकी है, बताइए। उत्तर:", 8, 0.0],
    ["hindi", "राम की उम्र, सीता की उम्र से कम है। सीता की उम्र, सुरेश की उम्र से कम है। राम और सुरेश में से किसकी उम्र ज़्यादा है? उत्तर:", 8, 0.0],
    ["nepali", "यदि विकासको उमेर 23 वर्ष छ र सीताको उमेर 23 वर्ष छ भने, कसको उमेर बढी होला? जवाफ:", 8, 0.0],
    ["nepali", "थाहा छ कि विकासको उमेर, रामको उमेर भन्दा बढी छ, र रामको उमेर, मनोजको उमेर भन्दा बढी छ। यीमध्ये सबैभन्दा कम उमेर कसको हो, भन्नुहोस्। जवाफ:", 8, 0.0],
]


class ModelStore:
    """Lazily loads and caches (model, tokenizer) pairs so switching between
    the 4 possible (language, variant) combinations in the UI doesn't
    reload from disk on every generate() call."""

    def __init__(self, ckpt_paths: dict):
        self.ckpt_paths = ckpt_paths  # {(lang, variant): path or None}
        self._cache = {}
        self._tokenizers = {}

    def available_choices(self):
        return sorted({lang for (lang, _variant), path in self.ckpt_paths.items() if path})

    def variants_for(self, lang: str):
        return sorted(v for (l, v), path in self.ckpt_paths.items() if l == lang and path)

    def _tokenizer(self, lang: str):
        if lang not in self._tokenizers:
            self._tokenizers[lang] = spm.SentencePieceProcessor(model_file=str(LANGS[lang]["spm"]))
        return self._tokenizers[lang]

    def get(self, lang: str, variant: str, device: str):
        key = (lang, variant)
        if key not in self._cache:
            path = self.ckpt_paths.get(key)
            if not path:
                raise ValueError(f"No checkpoint configured for {lang}/{variant} -- pass --{lang}_{variant} at startup")
            model_cfg = GPTConfig.from_yaml(LANGS[lang]["cfg"])
            model = GPT(model_cfg).to(device)
            load_checkpoint(path, model, map_location=device)
            model.eval()
            self._cache[key] = model
        return self._cache[key], self._tokenizer(lang)


def build_app(store: ModelStore, device: str):
    import gradio as gr

    lang_choices = store.available_choices()
    if not lang_choices:
        raise SystemExit("No checkpoints configured -- pass at least one of --hindi_pretrained/--hindi_finetuned/--nepali_pretrained/--nepali_finetuned")

    def generate_fn(lang, variant, prompt, max_new_tokens, temperature):
        if not prompt.strip():
            return "(enter a prompt)"
        model, sp = store.get(lang, variant, device)
        prompt_ids = sp.encode(prompt, out_type=int)
        idx = torch.tensor([prompt_ids], device=device)
        out = generate(model, idx, int(max_new_tokens), temperature=temperature or None, eos_id=sp.eos_id())
        gen_ids = out[0, len(prompt_ids):].tolist()
        if gen_ids and gen_ids[-1] == sp.eos_id():
            gen_ids = gen_ids[:-1]
        continuation = sp.decode(gen_ids)
        return prompt + continuation

    with gr.Blocks(title="LMA -- Model H / Model L test interface") as demo:
        gr.Markdown(
            "# Model H (Hindi) / Model L (Nepali) -- interactive test interface\n"
            "Pick a language and checkpoint variant, enter a prompt (a "
            "reasoning-style prompt ending in the answer cue works best for "
            "the finetuned checkpoints -- उत्तर: / जवाफ:), and generate."
        )
        with gr.Row():
            lang = gr.Radio(choices=lang_choices, value=lang_choices[0], label="Language")
            variant = gr.Radio(choices=store.variants_for(lang_choices[0]), value=store.variants_for(lang_choices[0])[0], label="Checkpoint")
        prompt = gr.Textbox(lines=3, label="Prompt")
        with gr.Row():
            max_new_tokens = gr.Slider(1, 128, value=32, step=1, label="Max new tokens")
            temperature = gr.Slider(0, 1.5, value=0.0, step=0.1, label="Temperature (0 = greedy)")
        run_btn = gr.Button("Generate", variant="primary")
        output = gr.Textbox(lines=6, label="Prompt + generated continuation")

        lang.change(lambda l: gr.Radio(choices=store.variants_for(l), value=store.variants_for(l)[0]), inputs=lang, outputs=variant)
        run_btn.click(generate_fn, inputs=[lang, variant, prompt, max_new_tokens, temperature], outputs=output)
        gr.Examples(examples=[e for e in EXAMPLES if e[0] in lang_choices], inputs=[lang, prompt, max_new_tokens, temperature])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hindi_pretrained", default=None)
    parser.add_argument("--hindi_finetuned", default=None)
    parser.add_argument("--nepali_pretrained", default=None)
    parser.add_argument("--nepali_finetuned", default=None)
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live link (needed on Colab)")
    args = parser.parse_args()

    ckpt_paths = {
        ("hindi", "pretrained"): args.hindi_pretrained,
        ("hindi", "finetuned"): args.hindi_finetuned,
        ("nepali", "pretrained"): args.nepali_pretrained,
        ("nepali", "finetuned"): args.nepali_finetuned,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = ModelStore(ckpt_paths)
    app = build_app(store, device)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
