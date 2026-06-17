"""Centralized helper for the LLM-as-judge grammatical-error annotator.

This is the *single source of truth* for the judge. It bundles three concerns,
kept as separate, independently testable functions:

    1. TAXONOMY SPEC      -- the label set, definitions, one canonical example per
                             category, and the coarse BLiMP-phenomenon map. The
                             judge prompt is rendered from this spec, so the prompt
                             and any future human-annotation guide stay identical.
    2. JUDGE CALL LOGIC   -- builds messages and calls either `ollama.chat` or
                             a local HuggingFace causal LM, then parses the reply
                             defensively.
    3. PIPELINE ADAPTER   -- maps a parsed result into the wide CSV row format used
                             by the rest of the eval pipeline (one binary column per
                             error category) plus an aggregate summary dict.

Design choices (locked in Phase A):
    - Taxonomy:   12 Hiller-Fernandez production-error labels (from
                  ``grammaticality.utils``) + a coarse ``judge_blimp_phenomenon``
                  field. Two routing labels: ``grammatical`` / ``unintelligible``.
    - Multi-label: an utterance may carry several error labels at once.
    - Output:     structured JSON from the model, parsed defensively.
    - Model:      ``qwen3:8b`` by default, swap via ``JudgeConfig.model``.

Nothing about the label set is hardcoded inside the call loop -- the loop reads it
from ``ERROR_LABELS`` / ``TAXONOMY`` below. To edit the taxonomy, edit this file
only.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from collections import OrderedDict

# ---------------------------------------------------------------------------
# 1. TAXONOMY SPEC  (single source of truth)
# ---------------------------------------------------------------------------
#
# The label *strings* originate in ``grammaticality.utils`` (the Hiller-Fernandez
# scheme already wired into the manual-annotation pipeline and the ``labels``
# field). We import them so the judge can never drift from that vocabulary. The
# import pulls heavy deps (enchant, matplotlib), so we fall back to literal copies
# of the same strings when the package is unavailable -- this keeps the taxonomy
# and adapter unit-testable offline. The fallback values are asserted equal to the
# imported ones whenever both are present (see ``_check_label_strings``).

try:  # pragma: no cover - exercised in the full project env
    from grammaticality.utils import (
        ERR_AUXILIARY,
        ERR_DETERMINER,
        ERR_OBJECT,
        ERR_OTHER,
        ERR_PLURAL,
        ERR_POSSESSIVE,
        ERR_PREPOSITION,
        ERR_PROGRESSIVE,
        ERR_SUBJECT,
        ERR_SV_AGREEMENT,
        ERR_TENSE_ASPECT,
        ERR_VERB,
    )

    _IMPORTED_LABELS = True
except Exception:  # ModuleNotFoundError or heavy-dep ImportError
    ERR_SUBJECT = "subject"
    ERR_VERB = "verb"
    ERR_OBJECT = "object"
    ERR_POSSESSIVE = "possessive"
    ERR_PLURAL = "plural"
    ERR_SV_AGREEMENT = "sv_agreement"
    ERR_TENSE_ASPECT = "tense_aspect"
    ERR_PROGRESSIVE = "progressive"
    ERR_DETERMINER = "determiner"
    ERR_PREPOSITION = "preposition"
    ERR_AUXILIARY = "auxiliary"
    ERR_OTHER = "other"
    _IMPORTED_LABELS = False

# Routing labels (not production-error categories).
GRAMMATICAL = "grammatical"  # no detectable grammatical error
UNINTELLIGIBLE = "unintelligible"  # non-word / babble / not analyzable
UNKNOWN = "unk"  # parser fallback when the model output is unusable


# Ordered taxonomy: label -> definition + one canonical example + coarse BLiMP
# phenomenon. Examples are drawn from the ERRORS_* banks in grammaticality.utils
# and written as "child form -> target form". Order is stable and drives the
# column order of the wide output.
TAXONOMY: "OrderedDict[str, dict]" = OrderedDict(
    [
        (
            ERR_SUBJECT,
            {
                "definition": "Wrong, missing, or miscased subject (often a subject pronoun in the wrong form).",
                "example": '"me want it" -> "I want it"',
                "blimp_phenomenon": "anaphor_agreement",
            },
        ),
        (
            ERR_VERB,
            {
                "definition": "Wrong main-verb form, or a missing/extra copula or lexical verb.",
                "example": '"it broken" -> "it is broken"',
                "blimp_phenomenon": "argument_structure",
            },
        ),
        (
            ERR_OBJECT,
            {
                "definition": "Wrong or miscased object (often an object pronoun in the wrong form).",
                "example": '"give it to she" -> "give it to her"',
                "blimp_phenomenon": "argument_structure",
            },
        ),
        (
            ERR_POSSESSIVE,
            {
                "definition": "Wrong or missing possessive marking ('s) or possessive pronoun.",
                "example": '"me ball" -> "my ball"',
                "blimp_phenomenon": "anaphor_agreement",
            },
        ),
        (
            ERR_PLURAL,
            {
                "definition": "Wrong noun number: over-regularized or missing plural marking.",
                "example": '"two foots" -> "two feet"',
                "blimp_phenomenon": "determiner_noun_agreement",
            },
        ),
        (
            ERR_SV_AGREEMENT,
            {
                "definition": "Subject-verb agreement error (person/number mismatch on the verb).",
                "example": '"he don\'t want" -> "he doesn\'t want"',
                "blimp_phenomenon": "subject_verb_agreement",
            },
        ),
        (
            ERR_TENSE_ASPECT,
            {
                "definition": "Wrong tense/aspect: over-regularized or incorrect past/perfect form.",
                "example": '"he goed home" -> "he went home"',
                "blimp_phenomenon": "irregular_forms",
            },
        ),
        (
            ERR_PROGRESSIVE,
            {
                "definition": "Progressive error: missing be-auxiliary and/or missing -ing.",
                "example": '"I go to play" -> "I\'m going to play"',
                "blimp_phenomenon": "irregular_forms",
            },
        ),
        (
            ERR_DETERMINER,
            {
                "definition": "Wrong or missing determiner/article (a/an/the).",
                "example": '"want apple" -> "want an apple"',
                "blimp_phenomenon": "determiner_noun_agreement",
            },
        ),
        (
            ERR_PREPOSITION,
            {
                "definition": "Wrong or missing preposition.",
                "example": '"look in me" -> "look at me"',
                "blimp_phenomenon": "argument_structure",
            },
        ),
        (
            ERR_AUXILIARY,
            {
                "definition": "Wrong or missing auxiliary/modal (do/does/have/will/can ...).",
                "example": '"you want it?" -> "do you want it?"',
                "blimp_phenomenon": "subject_verb_agreement",
            },
        ),
        (
            ERR_OTHER,
            {
                "definition": "A clear grammatical error that does not fit any category above "
                "(e.g. over-regularized comparative/superlative or reflexive).",
                "example": '"this one is gooder" -> "this one is better"',
                "blimp_phenomenon": "other",
            },
        ),
    ]
)

# The 12 production-error labels, in canonical order.
ERROR_LABELS: list[str] = list(TAXONOMY.keys())

# Every label the judge is allowed to emit (errors + routing labels).
ALL_LABELS: list[str] = ERROR_LABELS + [GRAMMATICAL, UNINTELLIGIBLE]

# Coarse BLiMP phenomenon for each error label (Q1: "H-F + coarse blimp field").
BLIMP_PHENOMENON_MAP: dict[str, str] = {
    label: spec["blimp_phenomenon"] for label, spec in TAXONOMY.items()
}


def _check_label_strings() -> None:
    """Assert the fallback label strings match the imported ones (run in full env)."""
    if not _IMPORTED_LABELS:
        return
    expected = {
        "subject": ERR_SUBJECT,
        "verb": ERR_VERB,
        "object": ERR_OBJECT,
        "possessive": ERR_POSSESSIVE,
        "plural": ERR_PLURAL,
        "sv_agreement": ERR_SV_AGREEMENT,
        "tense_aspect": ERR_TENSE_ASPECT,
        "progressive": ERR_PROGRESSIVE,
        "determiner": ERR_DETERMINER,
        "preposition": ERR_PREPOSITION,
        "auxiliary": ERR_AUXILIARY,
        "other": ERR_OTHER,
    }
    for literal, imported in expected.items():
        assert literal == imported, (
            f"Label string drift: judge expects '{literal}' but grammaticality.utils "
            f"defines '{imported}'. Reconcile the two."
        )


_check_label_strings()


def render_taxonomy_block() -> str:
    """Render the full taxonomy (labels + definitions + examples) as prompt text."""
    lines = ["GRAMMATICAL ERROR CATEGORIES:"]
    for i, (label, spec) in enumerate(TAXONOMY.items(), start=1):
        lines.append(f"{i:>2}. {label}: {spec['definition']}")
        lines.append(f"      example: {spec['example']}")
    lines.append("")
    lines.append("ROUTING LABELS (use exactly one of these instead of an error label "
                 "when applicable):")
    lines.append(f"  - {GRAMMATICAL}: the utterance has no detectable grammatical error.")
    lines.append(f"  - {UNINTELLIGIBLE}: the utterance is babble / non-words / cannot be "
                 "grammatically analyzed.")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a linguist annotating the grammatical errors in utterances produced by "
    "a model imitating a young child. You classify the grammatical-error type(s) of "
    "each utterance using a fixed taxonomy. Judge only grammar (morphology/syntax). "
    "Do NOT penalize childish phonological spellings or informal/dialectal contractions "
    "(e.g. 'wanna', 'gonna', 'doggie', 'wabbit'); those are not grammatical errors. "
    "An utterance may contain multiple errors -> return all that apply. "
    "Always respond with a single JSON object and nothing else."
)


def build_prompt(utterance: str, context: str | None = None) -> str:
    """Build the user prompt for one utterance, embedding the full taxonomy spec."""
    taxonomy_block = render_taxonomy_block()
    schema = (
        "Respond with ONLY a JSON object of this exact shape:\n"
        "{\n"
        '  "is_grammatical": true | false,\n'
        '  "labels": [<one or more labels from the taxonomy>],\n'
        '  "rationale": "<one short sentence>"\n'
        "}\n"
        "Rules:\n"
        f"  - Each label MUST be one of: {', '.join(ALL_LABELS)}.\n"
        '  - If there is no grammatical error, set is_grammatical=true and labels=["grammatical"].\n'
        '  - If the utterance is babble/non-words, set labels=["unintelligible"].\n'
        "  - Otherwise set is_grammatical=false and list every applicable error label."
    )
    parts = [taxonomy_block, ""]
    if context:
        parts.append(f'Preceding caregiver turn (context only): "{context}"')
    parts.append(f'Utterance to annotate: "{utterance}"')
    parts.append("")
    parts.append(schema)
    return "\n".join(parts)


def build_messages(utterance: str, context: str | None = None) -> list[dict]:
    """Build the chat messages list passed to ``ollama.chat``."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(utterance, context)},
    ]


# ---------------------------------------------------------------------------
# 1b. CHECKLIST MODE (Option C)  -- per-category true/false in one structured call
# ---------------------------------------------------------------------------
#
# Used by the few-shot benchmark (scripts/eval/benchmark_llm_judge_fewshot.py). The
# deployed free-form judge above is left untouched; these are parallel builders/
# parser for a different output contract: a single JSON object that maps EVERY
# category to a boolean. They are generic over a caller-supplied ``labels`` list +
# ``definitions`` map, so the benchmark can run them over the data's label space
# (which uses "present_progressive") without coupling this module to that spelling.

CHECKLIST_SYSTEM_PROMPT = (
    "You are a linguist annotating the grammatical errors in an utterance produced "
    "by a young child. The utterance is known to contain at least one grammatical "
    "error. For EACH category in the taxonomy you must decide whether an error of "
    "that category is present. Judge only grammar (morphology/syntax). Do NOT "
    "penalize childish phonological spellings or informal/dialectal contractions "
    "(e.g. 'wanna', 'gonna', 'doggie'); those are not grammatical errors. Multiple "
    "categories may be true. Respond with a single JSON object that maps every "
    "category name to true or false, and nothing else."
)


def render_checklist_definitions(labels: list[str], definitions: dict[str, str]) -> str:
    """Render the category definitions for checklist mode over ``labels``."""
    lines = ["GRAMMATICAL ERROR CATEGORIES (decide true/false for each):"]
    for i, label in enumerate(labels, start=1):
        lines.append(f"{i:>2}. {label}: {definitions.get(label, '')}")
    return "\n".join(lines)


def render_fewshot_block(examples: list[tuple[str, list[str]]], labels: list[str]) -> str:
    """Render demos as full per-category checklists.

    ``examples`` is a list of ``(utterance, true_labels)``. Each demo shows the
    complete JSON object (mostly ``false``) so the model is calibrated toward
    ``false`` by default and learns the exact output shape.
    """
    lines = ["WORKED EXAMPLES (each shows the full checklist for one utterance):", ""]
    for utterance, true_labels in examples:
        truth = set(true_labels)
        obj = {label: (label in truth) for label in labels}
        lines.append(f'Utterance: "{utterance}"')
        lines.append(json.dumps(obj))
        lines.append("")
    return "\n".join(lines)


def build_checklist_messages(
    utterance: str,
    labels: list[str],
    definitions: dict[str, str],
    fewshot_block: str,
) -> list[dict]:
    """Build the chat messages for one checklist judgment (Option C)."""
    schema = (
        "OUTPUT: a single JSON object whose keys are EXACTLY these category names: "
        + ", ".join(labels)
        + ". Each value must be true or false. Output JSON only, no prose."
    )
    user = "\n".join(
        [
            render_checklist_definitions(labels, definitions),
            "",
            fewshot_block,
            schema,
            "",
            f'Utterance to annotate: "{utterance}"',
        ]
    )
    return [
        {"role": "system", "content": CHECKLIST_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_checklist_json(
    raw: str, labels: list[str], aliases: dict[str, str] | None = None
) -> tuple[dict[str, int], bool]:
    """Defensively parse a per-category boolean object.

    Returns ``(preds, ok)`` where ``preds`` maps every label in ``labels`` to 0/1
    (defaulting to 0 for missing keys) and ``ok`` is False if nothing parseable was
    found. Keys are normalized + alias-resolved so e.g. "progressive" maps onto
    "present_progressive".
    """
    aliases = aliases or {}
    preds = {label: 0 for label in labels}
    label_set = set(labels)

    text = (raw or "").strip()
    if not text:
        return preds, False
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return preds, False
    if not isinstance(data, dict):
        return preds, False

    for key, value in data.items():
        norm = str(key).strip().lower().replace(" ", "_").replace("-", "_")
        norm = aliases.get(norm, norm)
        if norm in label_set:
            truthy = value is True or str(value).strip().lower() in {"true", "1", "yes"}
            preds[norm] = 1 if truthy else 0
    return preds, True


# ---------------------------------------------------------------------------
# 2. OLLAMA CALL LOGIC
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class JudgeConfig:
    """Runtime configuration for the Ollama judge. Edit ``model`` to swap models."""

    model: str = "qwen3:8b"
    host: str | None = None  # None -> ollama default (http://localhost:11434)
    temperature: float = 0.0
    max_retries: int = 3
    retry_backoff: float = 2.0  # seconds, multiplied each retry
    disable_thinking: bool = True  # qwen3 etc. emit <think>...; ask for direct JSON
    request_timeout: float | None = None


@dataclasses.dataclass
class JudgeResult:
    """One judged utterance, before adaptation to the pipeline row format."""

    utterance: str
    labels: list[str]
    is_grammatical: bool | None
    rationale: str
    raw: str
    model: str
    ok: bool  # False => parsing/calling failed, labels fell back to ["unk"]


def _get_client(config: JudgeConfig):
    """Lazily import ollama and return a client bound to the configured host."""
    import ollama  # local import: not needed for taxonomy/adapter unit tests

    if config.host:
        return ollama.Client(host=config.host)
    return ollama


def call_judge_once(messages: list[dict], config: JudgeConfig, client=None) -> str:
    """Single Ollama chat call in JSON mode. Returns the raw message content."""
    client = client or _get_client(config)
    options = {"temperature": config.temperature}

    kwargs = dict(model=config.model, messages=messages, format="json", options=options)
    # ``think`` is only supported by newer ollama clients; degrade gracefully.
    if config.disable_thinking:
        try:
            response = client.chat(think=False, **kwargs)
        except TypeError:
            response = client.chat(**kwargs)
    else:
        response = client.chat(**kwargs)
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# 2b. HUGGINGFACE CALL LOGIC
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class HFJudgeConfig:
    """Runtime configuration for a HuggingFace local judge model."""

    model: str = "Qwen/Qwen3-8B"
    dtype: str = "bfloat16"
    device: str = "auto"
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    enable_thinking: bool = False
    batch_size: int = 16
    max_retries: int = 2
    retry_backoff: float = 2.0
    retry_do_sample: bool = True
    retry_temperature: float = 0.3
    trust_remote_code: bool = False


def _version_tuple(version: str) -> tuple[int, int, int]:
    """Return a coarse comparable version tuple without importing packaging."""
    parts = []
    for part in version.split(".")[:3]:
        digits = ""
        for char in part:
            if char.isdigit():
                digits += char
            else:
                break
        parts.append(int(digits or 0))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _resolve_torch_dtype(dtype: str, torch) -> object:
    """Map a CLI dtype string to the value expected by transformers."""
    normalized = (dtype or "auto").lower()
    if normalized == "auto":
        return "auto"
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
        "float": "float32",
    }
    attr = aliases.get(normalized)
    if attr is None or not hasattr(torch, attr):
        raise ValueError(f"Unsupported HF dtype '{dtype}'. Use auto, bfloat16, float16, or float32.")
    return getattr(torch, attr)


def load_hf_judge(config: HFJudgeConfig) -> tuple[object, object]:
    """Load a HuggingFace causal LM judge once and return ``(model, tokenizer)``."""
    import torch  # local import: not needed for taxonomy/adapter unit tests
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config.model.lower().startswith("qwen/qwen3") and _version_tuple(transformers.__version__) < (4, 51, 0):
        raise RuntimeError(
            f"{config.model} requires transformers>=4.51.0; found {transformers.__version__}. "
            "Use the newer cluster environment before running the HF judge."
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": _resolve_torch_dtype(config.dtype, torch),
        "trust_remote_code": config.trust_remote_code,
    }
    if config.device == "auto":
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.model, **load_kwargs)
    if config.device != "auto":
        model = model.to(config.device)
    model.eval()
    return model, tokenizer


def _format_hf_chat(messages: list[dict], tokenizer, config: HFJudgeConfig) -> str:
    """Render chat messages with the model's tokenizer chat template."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.enable_thinking,
        )
    except TypeError:
        if config.enable_thinking:
            raise
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _strip_thinking_blocks(text: str) -> str:
    """Remove Qwen-style thinking blocks before JSON parsing."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1].strip()
    if cleaned.startswith("<think>"):
        start = cleaned.find("{")
        if start != -1:
            cleaned = cleaned[start:]
    return cleaned.strip()


def _hf_generation_kwargs(config: HFJudgeConfig, tokenizer) -> dict:
    """Build kwargs for ``model.generate``."""
    kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if config.do_sample:
        if config.temperature > 0:
            kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.top_k is not None:
            kwargs["top_k"] = config.top_k
    return kwargs


def call_judge_hf_batch(messages_batch: list[list[dict]], model, tokenizer, config: HFJudgeConfig) -> list[str]:
    """Call a HuggingFace judge for a batch of chat messages."""
    import torch  # local import: not needed for taxonomy/adapter unit tests

    if not messages_batch:
        return []

    texts = [_format_hf_chat(messages, tokenizer, config) for messages in messages_batch]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True)
    model_device = getattr(model, "device", None)
    if model_device is not None:
        model_inputs = model_inputs.to(model_device)

    input_len = model_inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(**model_inputs, **_hf_generation_kwargs(config, tokenizer))

    raw_outputs = []
    for output_ids in generated[:, input_len:]:
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        raw_outputs.append(_strip_thinking_blocks(text))
    return raw_outputs


def call_judge_hf(messages: list[dict], model, tokenizer, config: HFJudgeConfig) -> str:
    """Call a HuggingFace judge for one chat message list."""
    return call_judge_hf_batch([messages], model, tokenizer, config)[0]


def parse_judge_json(raw: str) -> dict:
    """Defensively parse the model's JSON reply.

    Returns a dict with keys ``labels`` (list[str]), ``is_grammatical`` (bool|None),
    ``rationale`` (str). Raises ``ValueError`` if nothing usable can be extracted.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty response")

    # Strip a leading ```json fence if present, then isolate the outermost object.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    data = json.loads(text)  # may raise json.JSONDecodeError (subclass of ValueError)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON is not an object")

    raw_labels = data.get("labels", [])
    if isinstance(raw_labels, str):
        raw_labels = [raw_labels]
    elif not isinstance(raw_labels, list):
        raw_labels = []

    # Normalize, keep only known labels, dedupe preserving order.
    labels: list[str] = []
    for lab in raw_labels:
        norm = str(lab).strip().lower().replace(" ", "_").replace("-", "_")
        if norm in ALL_LABELS and norm not in labels:
            labels.append(norm)

    is_grammatical = data.get("is_grammatical", None)
    if not isinstance(is_grammatical, bool):
        is_grammatical = None

    rationale = str(data.get("rationale", "")).strip()
    return {"labels": labels, "is_grammatical": is_grammatical, "rationale": rationale}


def _reconcile_labels(labels: list[str], is_grammatical: bool | None) -> tuple[list[str], bool | None]:
    """Make labels and the grammaticality flag mutually consistent."""
    has_error = any(lab in ERROR_LABELS for lab in labels)
    if UNINTELLIGIBLE in labels:
        return [UNINTELLIGIBLE], None
    if has_error:
        # Drop the contradictory "grammatical" routing label if present.
        labels = [lab for lab in labels if lab != GRAMMATICAL]
        return labels, False
    # No error labels -> treat as grammatical.
    return [GRAMMATICAL], True


def judge_utterance(utterance: str, config: JudgeConfig, context: str | None = None, client=None) -> JudgeResult:
    """Judge a single utterance with retries; never raises -- falls back to ``unk``."""
    text = (utterance or "").strip()
    if not text:
        # Empty rows are kept (Q4/Q6: rows are never dropped) and routed out.
        return JudgeResult(
            utterance=utterance,
            labels=[UNINTELLIGIBLE],
            is_grammatical=None,
            rationale="empty utterance",
            raw="",
            model=config.model,
            ok=True,
        )

    client = client or _get_client(config)
    messages = build_messages(text, context)
    last_raw = ""
    last_err = ""
    for attempt in range(1, config.max_retries + 1):
        try:
            last_raw = call_judge_once(messages, config, client=client)
            parsed = parse_judge_json(last_raw)
            labels, is_gram = _reconcile_labels(parsed["labels"], parsed["is_grammatical"])
            return JudgeResult(
                utterance=utterance,
                labels=labels,
                is_grammatical=is_gram,
                rationale=parsed["rationale"],
                raw=last_raw,
                model=config.model,
                ok=True,
            )
        except Exception as exc:  # noqa: BLE001 - judge must be robust
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < config.max_retries:
                time.sleep(config.retry_backoff * attempt)

    # All retries exhausted -> graceful fallback, logged via the raw/rationale fields.
    return JudgeResult(
        utterance=utterance,
        labels=[UNKNOWN],
        is_grammatical=None,
        rationale=f"parse/call failed: {last_err}",
        raw=last_raw,
        model=config.model,
        ok=False,
    )


# ---------------------------------------------------------------------------
# 3. PIPELINE ADAPTER  (-> wide CSV row format)
# ---------------------------------------------------------------------------
#
# Output columns appended to the input CSV (Q4 wide format). One binary column per
# error category, plus a grammaticality flag, the multi-label string, the coarse
# BLiMP phenomenon, and audit fields for the kappa validation hook (Q6).

_COL_PREFIX = "judge_"

# Stable, ordered list of columns this method appends. Used by the CLI to keep
# column order deterministic and to know which columns to (re)write.
JUDGE_COLUMNS: list[str] = (
    [f"{_COL_PREFIX}{label}" for label in ERROR_LABELS]  # multi-hot error columns
    + [
        f"{_COL_PREFIX}is_grammatical",  # 1 grammatical / 0 not / "" unintelligible
        f"{_COL_PREFIX}labels",  # comma-joined labels (matches existing `labels`)
        f"{_COL_PREFIX}blimp_phenomenon",  # coarse BLiMP phenomenon(s)
        f"{_COL_PREFIX}rationale",  # audit
        f"{_COL_PREFIX}raw",  # audit: raw model output
        f"{_COL_PREFIX}model",  # audit: which judge model produced this
        f"{_COL_PREFIX}ok",  # audit: did parsing succeed
    ]
)


def result_to_row(result: JudgeResult) -> dict:
    """Map a :class:`JudgeResult` to the wide pipeline row (dict of JUDGE_COLUMNS)."""
    row: dict = {f"{_COL_PREFIX}{label}": 0 for label in ERROR_LABELS}
    for lab in result.labels:
        if lab in ERROR_LABELS:
            row[f"{_COL_PREFIX}{lab}"] = 1

    if UNINTELLIGIBLE in result.labels or result.is_grammatical is None:
        gram_flag: object = ""  # not applicable / unknown
    else:
        gram_flag = int(bool(result.is_grammatical))

    phenomena = [BLIMP_PHENOMENON_MAP[lab] for lab in result.labels if lab in BLIMP_PHENOMENON_MAP]
    # dedupe, preserve order
    phenomena = list(dict.fromkeys(phenomena))

    row[f"{_COL_PREFIX}is_grammatical"] = gram_flag
    row[f"{_COL_PREFIX}labels"] = ", ".join(result.labels)
    row[f"{_COL_PREFIX}blimp_phenomenon"] = ", ".join(phenomena)
    row[f"{_COL_PREFIX}rationale"] = result.rationale
    row[f"{_COL_PREFIX}raw"] = result.raw
    row[f"{_COL_PREFIX}model"] = result.model
    row[f"{_COL_PREFIX}ok"] = int(result.ok)
    return row


def aggregate_summary(results: list[JudgeResult]) -> dict:
    """Build the scalar summary dict (additive keys for the aggregate results CSV)."""
    n = len(results)
    summary: dict = {"judge_num_scored": n, "judge_num_malformed": sum(not r.ok for r in results)}
    if n == 0:
        return summary

    # Per-label prevalence (fraction of utterances carrying each error label).
    for label in ERROR_LABELS:
        count = sum(label in r.labels for r in results)
        summary[f"judge_rate_{label}"] = count / n

    grammatical = sum(r.is_grammatical is True for r in results)
    unintelligible = sum(UNINTELLIGIBLE in r.labels for r in results)
    summary["judge_rate_grammatical"] = grammatical / n
    summary["judge_rate_unintelligible"] = unintelligible / n
    return summary
