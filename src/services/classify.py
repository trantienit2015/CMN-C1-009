"""Deterministic classification domain logic for CMN-C1-009 ClassificationAgent.

Input validation, text normalization, taxonomy validation, prompt building, and
LLM-response parsing — the non-framework logic lifted verbatim from the original
node implementations (InputNormalize / TaxonomyLoad / LLMClassify). Pure
functions, no state, no external imports beyond stdlib.
"""

from __future__ import annotations

from typing import Any
import json
import re

VALID_MODES = {"binary", "multi-class", "multi-label"}
DEFAULT_MAX_LEN = 8000
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# S-2: prompt-injection / credential patterns scanned in the raw input.
INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore\s+(previous|all|above|prior)\s+(instructions?|prompts?|context)"),
    re.compile(r"(?i)you\s+are\s+now\s+"),
    re.compile(r"(?i)(system\s*:|\[INST\]|<\|im_start\|>)"),
    re.compile(r"(?i)disregard\s+(your|all)"),
    re.compile(r"(?i)jailbreak"),
]

CREDENTIAL_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"eyJ[a-zA-Z0-9._\-]{10,}"),
    re.compile(r"(?i)(password|api_key|token|secret)\s*[=:]\s*[\x22\x27][^\x22\x27]{8,}[\x22\x27]"),
    re.compile(r"Bearer\s+[a-zA-Z0-9._\-]{20,}"),
]

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_MARKDOWN_FENCE = re.compile(r"^```[a-z]*\s*|```\s*$", flags=re.MULTILINE)


def validate_input(input_text: Any, classification_mode: Any, max_input_length: Any) -> tuple[str, str] | None:
    """Return (error_code, error_message) if the input is invalid, else None.

    S-5: fail-fast on empty / over-length / unknown-mode input.
    S-2: reject credential patterns in the raw input.
    """
    if not isinstance(input_text, str) or not input_text.strip():
        return ("S5_EMPTY_INPUT", "input_text must be a non-empty string.")

    max_len = int(max_input_length or DEFAULT_MAX_LEN)
    if len(input_text) > max_len:
        return ("S5_INPUT_TOO_LONG", f"input_text exceeds max_input_length ({len(input_text)} > {max_len}).")

    for pat in CREDENTIAL_PATTERNS:
        if pat.search(input_text):
            return ("S2_CREDENTIAL_DETECTED", "Credential pattern detected in input_text.")

    if classification_mode not in VALID_MODES:
        return (
            "S5_INVALID_MODE",
            f"classification_mode must be one of {sorted(VALID_MODES)}, got {classification_mode!r}.",
        )

    return None


def normalize_text(input_text: str) -> str:
    """Strip HTML tags, collapse whitespace runs, trim — injection-guarded clean."""
    cleaned = _HTML_TAG.sub("", input_text)
    cleaned = _WHITESPACE.sub(" ", cleaned)
    return cleaned.strip()


def validate_taxonomy(label_list: Any, classification_mode: Any, few_shot_examples: Any) -> tuple[str, str] | None:
    """Validate the caller-supplied taxonomy and few-shot examples.

    S-2: no external fetch — taxonomy is always caller-provided. Returns
    (error_code, error_message) on rejection, else None.
    """
    if not label_list or not isinstance(label_list, list):
        return ("TAXONOMY_EMPTY", "label_list must be a non-empty list.")

    for i, label in enumerate(label_list):
        if not isinstance(label, str) or not label.strip():
            return ("TAXONOMY_INVALID_LABEL", f"label_list[{i}] must be a non-empty string.")

    if len(set(label_list)) != len(label_list):
        return ("TAXONOMY_DUPLICATE", "label_list contains duplicate labels.")

    if classification_mode == "binary" and len(label_list) != 2:
        return ("TAXONOMY_BINARY_ARITY", f"binary mode requires exactly 2 labels, got {len(label_list)}.")

    if few_shot_examples is not None:
        if not isinstance(few_shot_examples, list):
            return ("FEWSHOT_INVALID", "few_shot_examples must be a list.")
        label_set = set(label_list)
        for i, ex in enumerate(few_shot_examples):
            if not isinstance(ex, dict):
                return ("FEWSHOT_INVALID", f"few_shot_examples[{i}] must be a dict.")
            if not isinstance(ex.get("input"), str):
                return ("FEWSHOT_INVALID", f"few_shot_examples[{i}].input must be a string.")
            ex_labels = ex.get("labels")
            if not isinstance(ex_labels, list):
                return ("FEWSHOT_INVALID", f"few_shot_examples[{i}].labels must be a list.")
            for lbl in ex_labels:
                if lbl not in label_set:
                    return ("FEWSHOT_UNKNOWN_LABEL", f"few_shot_examples[{i}] contains unknown label: {lbl!r}.")

    return None


def build_prompt(normalized_text: str, label_list: list[Any], mode: str, few_shot: list[Any] | None) -> str:
    """Build a mode-specific classification prompt."""
    mode_instruction = {
        "binary": "Answer YES or NO only. Select exactly one label from the list.",
        "multi-class": "Select exactly one label from the list that best fits the input.",
        "multi-label": "Select all applicable labels from the list.",
    }.get(mode, "Select the most appropriate label(s) from the list.")

    label_str = ", ".join(f'"{lbl}"' for lbl in label_list)
    prompt_parts = [
        f"You are a text classification system. {mode_instruction}",
        f"\nAvailable labels: [{label_str}]",
    ]

    if few_shot:
        prompt_parts.append("\nExamples:")
        for ex in few_shot:
            ex_labels = ", ".join(f'"{lbl}"' for lbl in ex.get("labels", []))
            prompt_parts.append(f'  Input: {ex["input"]!r}\n  Output: {{"labels": [{ex_labels}], "confidence": 0.9}}')

    prompt_parts.append(
        f"\nInput to classify: {normalized_text!r}"
        f"\n\nRespond ONLY with a JSON object in this exact format:"
        f'\n{{"labels": [<label strings>], "confidence": <float 0.0-1.0>}}'
    )
    return "\n".join(prompt_parts)


def parse_response(raw: str, label_list: list[Any]) -> tuple[list[Any] | None, float | None, str | None]:
    """Parse and validate the LLM response.

    Returns (labels, confidence, error_message). Defensive: strips markdown
    fences, attempts JSON parse, falls back to substring label search.
    """
    label_set = set(label_list)
    cleaned = _MARKDOWN_FENCE.sub("", raw.strip()).strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    if parsed and isinstance(parsed, dict):
        raw_labels = parsed.get("labels", [])
        confidence = parsed.get("confidence")
        if isinstance(raw_labels, list):
            unknown = [lbl for lbl in raw_labels if lbl not in label_set]
            if unknown:
                return None, None, f"LLM returned unknown label(s): {unknown}"
            valid_labels = [lbl for lbl in raw_labels if isinstance(lbl, str) and lbl in label_set]
            if valid_labels:
                conf = float(confidence) if isinstance(confidence, (int, float)) else 0.5
                return valid_labels, conf, None

    matched = [lbl for lbl in label_list if lbl in raw]
    if matched:
        return matched, 0.5, None

    return None, None, "LLM output could not be parsed into valid labels"


def deterministic_fallback(label_list: list[Any], mode: str) -> tuple[list[Any], float]:
    """No-LLM-provisioned degrade mode: pick a label deterministically.

    Not a real classification — confidence 0.0 marks it unambiguously so a
    caller cannot mistake this for a model-scored result. Shape matches what
    `validate_output` expects per mode (exactly 1 for binary/multi-class, >=1
    for multi-label), so the rest of the pipeline (post_process, taxonomy
    checks) runs unchanged on this path.
    """
    if not label_list:
        return [], 0.0
    if mode == "multi-label":
        return [label_list[0]], 0.0
    return [label_list[0]], 0.0


def validate_output(labels: Any, confidence: Any, mode: Any, label_list: Any, threshold: Any) -> tuple[str, str] | None:
    """S-3 taxonomy-allowlist + mode-arity validation of the LLM labels.

    Returns (error_code, error_message) on rejection, else None.
    """
    if not labels or not isinstance(labels, list):
        return ("OUTPUT_EMPTY", "Output labels are empty or invalid.")

    if mode in ("binary", "multi-class"):
        if len(labels) != 1:
            return ("OUTPUT_ARITY", f"{mode} mode must produce exactly 1 label, got {len(labels)}.")
    elif mode == "multi-label":
        if len(labels) < 1:
            return ("OUTPUT_ARITY", "multi-label mode must produce at least 1 label.")

    disallowed = [lbl for lbl in labels if lbl not in set(label_list)]
    if disallowed:
        return ("S3_LABEL_NOT_IN_TAXONOMY", f"Output contains labels not in allowed taxonomy: {disallowed}.")

    return None


def format_result(labels: list[Any], confidence: float, mode: str, review_required: bool, threshold: float) -> str:
    """Render the classification result as human-readable Markdown.

    Separate from the JSON in ``final_output``: a parent Cat 2 agent consumes
    the JSON contract (docs/02 §8), while this string is what
    ``AgentBaseGraph.get_output()`` surfaces as ``output`` — i.e. what a person
    reads in the chat. Raw JSON there is parseable but not useful to a reader.

    Carries no label text the caller did not supply and adds no interpretation
    beyond the threshold comparison already recorded in ``review_required``.
    """
    label_line = ", ".join(f"`{lbl}`" for lbl in labels) if labels else "_none_"
    pct = f"{float(confidence) * 100:.0f}%"

    lines = [
        "**Classification result**",
        "",
        f"- **Label{'s' if len(labels) > 1 else ''}:** {label_line}",
        f"- **Confidence:** {pct}",
        f"- **Mode:** {mode}",
    ]
    if review_required:
        lines.append(
            f"- **Human review:** recommended — confidence {pct} is below the "
            f"{float(threshold) * 100:.0f}% threshold"
        )
    else:
        lines.append(
            f"- **Human review:** not required — confidence is at or above the "
            f"{float(threshold) * 100:.0f}% threshold"
        )
    return "\n".join(lines)
